// Inspired by TRT-LLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dispatch_utils.h"
#include "utils.cuh"
#include "reduction_utils.cuh"


namespace vllm {

// from TRTLLM
template <typename Tf, typename T>
__inline__ __device__ Tf compute_layernorm(Tf val, float s_mean, float s_variance, const T* gamma, const T* beta, int i)
{
    Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gamma[i]);
    if (beta != nullptr)
    {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

// from TRTLLM
/* Computes the layernorm https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 * normed_output <- ( (input - E[input]) / Sqrt(Var[input] + eps) ) * gamma + beta
 * input is [tokens, hidden_dim]. Mean and Variance are per-row (i.e. per-token)
 *
 * One CTA handles one row.
 *
 * with USE_DIFF_OF_SQUARES set to false:
 * First pass (loop) computes the mean.
 * Second computes the variance via Var[x] = E[(x - E[x])²].
 * Third pass computes and writes normed_output
 *
 * with USE_DIFF_OF_SQUARES set to true (may be faster but less accurate):
 * First pass (loop) computes the mean and variance via Var[x] = E[x²] - E[x]²
 * Second pass computes and writes normed_output
 *
 * use_shmem controls if we cache input values into shared memory
 *
 * Optional: with dynamic scaling, the last pass doesn't write immediately but finds the
 *           amax per row. A final pass scales to int8 accordingly, and writes output to
 *           normed_output_quant.
 */
template <typename T, typename scale_type, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm(const T* input, const T* gamma, const T* beta, T* normed_output, const float eps,
    int tokens, int hidden_dim, const scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
        }
    }
}


template <typename T, typename scale_type, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm_fuse_sum(const T* input, const T* gamma, const T* beta, T* normed_output, const float eps,
    int tokens, int hidden_dim, scale_type* input_sum, const scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;
    T_scalar sum = 0.0f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            sum += cuda_sum<float>(val);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        float sum_f = blockAllReduceSum(cuda_cast<float>(sum));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
            input_sum[bidx] = sum_f;
        }
    }
}


// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, typename out_type, bool use_quant>
__global__ void
rms_norm_kernel(out_type *__restrict__ out,         // [..., hidden_size]
                const scalar_t *__restrict__ input, // [..., hidden_size]
                const scalar_t *__restrict__ weight, // [hidden_size]
                const float epsilon, const int num_tokens,
                const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    if constexpr (use_quant) {
      out[blockIdx.x * hidden_size + idx] = float_to_int8_rn(
        ((float)(x * s_variance)) * (float)(weight[idx]));
    } else {
      out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
    }
  }
}




template <typename T, typename scale_type, bool use_per_token_dequant>
__global__ void dequant_add_residual_rms_norm_quant_kernel(
    const int32_t *__restrict__ input, T *__restrict__ residual,
    int8_t *__restrict__ output, const T *__restrict__ gamma,
    const float layernorm_eps, const scale_type scale, int num_tokens, int hidden_size) {
  // layernorm module in the T5 style No bias and no subtraction of mean.
  const int tid = threadIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;

  float local_var_sum = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float diff = 0.0f;
    if constexpr (use_per_token_dequant) {
      diff = ((((float)input[blockIdx.x * hidden_size + i]) * __half2float(scale[blockIdx.x])) +
                  (float)residual[blockIdx.x * hidden_size + i]);
    } else {
      diff = ((((float)input[blockIdx.x * hidden_size + i]) * __half2float(scale)) +
                  (float)residual[blockIdx.x * hidden_size + i]);
    }
    residual[blockIdx.x * hidden_size + i] = (T)diff;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum(local_var_sum);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / (float)hidden_size + layernorm_eps);
  }
  __syncthreads();

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    output[blockIdx.x * hidden_size + i] = float_to_int8_rn(
        (((float)(residual[blockIdx.x * hidden_size + i])) * s_variance) *
        (float)(gamma[i]));
  }
}
} // namespace vllm

void rms_norm(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              float epsilon,
              bool use_quant) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    if (use_quant) {
      vllm::rms_norm_kernel<scalar_t, int8_t, true><<<grid, block, 0, stream>>>(
        out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    } else {
      vllm::rms_norm_kernel<scalar_t, scalar_t, false><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    }
  });
}

void rms_norm_general(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon,
              bool use_per_token_quant) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  block.x = 32 * ((block.x + 31) / 32);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generalLayerNorm", [&] {
    using T = typename FloatTypeConverter<scalar_t>::Type;
    if (use_per_token_quant) {
      // per-token
      vllm::generalLayerNorm<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, nullptr, scaling.data_ptr<at::Half>(),
        out.data_ptr<int8_t>(), false
      );
      // input, gamma, beta, normed_output, eps, tokens, hidden_dim, per_tensor_scale, per_token_scale
      // normed_output_quant, use_shmem
        // out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
        // weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    } else {
      // per-tensor
      vllm::generalLayerNorm<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, scaling.data_ptr<at::Half>(), nullptr,
        out.data_ptr<int8_t>(), false
      );
    }
  });
}

void rms_norm_general_fuse_sum(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &input_sum, // [tokens] or [1]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon,
              bool use_per_token_quant) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  block.x = 32 * ((block.x + 31) / 32);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generalLayerNorm_fuse_sum", [&] {
    using T = typename FloatTypeConverter<scalar_t>::Type;
    if (use_per_token_quant) {
      // per-token
      vllm::generalLayerNorm_fuse_sum<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, input_sum.data_ptr<at::Half>(), nullptr, scaling.data_ptr<at::Half>(),
        out.data_ptr<int8_t>(), false
      );
      // input, gamma, beta, normed_output, eps, tokens, hidden_dim, per_tensor_scale, per_token_scale
      // normed_output_quant, use_shmem
        // out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
        // weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    } else {
      // per-tensor
      // Rasing error here
      // Not implemented per-tensor input_sum
      assert(false);
      
      vllm::generalLayerNorm_fuse_sum<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, nullptr, scaling.data_ptr<at::Half>(), nullptr,
        out.data_ptr<int8_t>(), false
      );
    }
  });
}



void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    at::Half scale,
    float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_rms_norm_quant_kernel",
      [&] {
          vllm::dequant_add_residual_rms_norm_quant_kernel<scalar_t, at::Half, false>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<int32_t>(), residual.data_ptr<scalar_t>(),
                out.data_ptr<int8_t>(), gamma.data_ptr<scalar_t>(), epsilon,
                scale, num_tokens, hidden_size);
      });
}

void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    torch::Tensor &scale,    // [num_tokens]
    float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_rms_norm_quant_kernel",
      [&] {
          vllm::dequant_add_residual_rms_norm_quant_kernel<scalar_t, at::Half*, true>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<int32_t>(), residual.data_ptr<scalar_t>(),
                out.data_ptr<int8_t>(), gamma.data_ptr<scalar_t>(), epsilon,
                scale.data_ptr<at::Half>(), num_tokens, hidden_size);
      });
}
