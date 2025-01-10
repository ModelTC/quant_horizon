from typing import Optional, Tuple, Union
import torch
import int8_cuda
from ..cutlass_w8a8 import *
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY
def is_hip() -> bool:
    return torch.version.hip is not None

TORCH_SCALED_MM_SCALE_RESULT = torch.ones(1).cuda() if is_hip() else None


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp is
            None), "azp must only be provided for asymmetric quantization."
        int8_cuda.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    int8_cuda.dynamic_scaled_int8_quant(output, input, input_scales,
                                           input_azp)
    return output, input_scales, input_azp

def apply_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_zero_point: Optional[torch.Tensor] = None,
    azp_adj: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    # ops.scaled_int8_quant supports both dynamic and static quant.
    # * dynamic, layer.input_scale is None and x_scale computed from x.
    # * static, layer.input_scale is scalar and x_scale is input_scale.
    symmetric = azp_adj is None
    x_q, x_scale, x_zp = scaled_int8_quant(input,
                                               input_scale,
                                               input_zero_point,
                                               symmetric=symmetric)

    if x_zp is not None:
        # Currently, static is always per-tensor and dynamic is per-token
        static = input_zero_point is not None
        azp = None if static else x_zp
        return cutlass_scaled_mm_azp(x_q,
                                         weight,
                                         scale_a=x_scale,
                                         scale_b=weight_scale,
                                         out_dtype=input.dtype,
                                         azp_adj=azp_adj,
                                         azp=azp,
                                         bias=bias)
    return cutlass_scaled_mm(x_q,
                                 weight,
                                 scale_a=x_scale,
                                 scale_b=weight_scale,
                                 out_dtype=input.dtype,
                                 bias=bias)

def init_int8_static(A_shape, B_shape, A_data, B_data):

    A_data_int8, scale_A, zp_A = scaled_int8_quant(A_data)
    B_data_int8, scale_B, zp_B = scaled_int8_quant(B_data.t().contiguous())
    B_data_int8 = B_data_int8.contiguous().t()

    return {"A_data_int8":A_data_int8, "scale_A":scale_A, "B_data_int8":B_data_int8, "scale_B":scale_B}

def run_int8_static(A_data_int8, scale_A, B_data_int8, scale_B):
    
    cutlass_scaled_mm(
        A_data_int8,
        B_data_int8,
        scale_a=scale_A,
        scale_b=scale_B,
        out_dtype=torch.float16
    )
    
def get_int8_static_res(A_data_int8, scale_A, B_data_int8, scale_B):
    C_fp16 = cutlass_scaled_mm(
        A_data_int8,
        B_data_int8,
        scale_a=scale_A,
        scale_b=scale_B,
        out_dtype=torch.float16
    )
    return C_fp16
   

def init_int8_dynamic(A_shape, B_shape, A_data, B_data,):
    B_data_int8, scale_B, zp_B = scaled_int8_quant(B_data.t().contiguous())
    B_data_int8 = B_data_int8.contiguous().t()
    return {"A_data":A_data, "B_data_int8":B_data_int8, "scale_B":scale_B}

def run_int8_dynamic(A_data, B_data_int8, scale_B):
    A_data_int8, scale_A, zp_A = scaled_int8_quant(A_data)
    cutlass_scaled_mm(
        A_data_int8,
        B_data_int8,
        scale_a=scale_A,
        scale_b=scale_B,
        out_dtype=torch.float16
    )
    
def get_int8_dynamic_res(A_data, B_data_int8, scale_B):
    A_data_int8, scale_A, zp_A = scaled_int8_quant(A_data)
    C_fp16 = cutlass_scaled_mm(
        A_data_int8,
        B_data_int8,
        scale_a=scale_A,
        scale_b=scale_B,
        out_dtype=torch.float16
    )
    return C_fp16
   

SPEED_REGISTRY.register("int8_static", run_int8_static, init_int8_static)
ACC_REGISTRY.register("int8_static", get_int8_static_res, init_int8_static)

SPEED_REGISTRY.register("int8_dynamic", run_int8_dynamic, init_int8_dynamic)
ACC_REGISTRY.register("int8_dynamic", get_int8_dynamic_res, init_int8_dynamic)


if __name__ == "__main__":
    A_shape = (16, 4096)
    B_shape = (4096, 11008)
    A_data = torch.randn(A_shape[0], A_shape[1], dtype=torch.float16, device="cuda")
    B_data = torch.randn(B_shape[0], B_shape[1], dtype=torch.float16, device="cuda")
    init_params = {
        "default": {
            "A_shape": A_shape,
            "B_shape": B_shape,
            "A_data": A_data,
            "B_data": B_data,
        },
        "int8_static": {},
        "int8_dynamic": {},
    }

    SPEED_REGISTRY.benchmark_all(init_params)
    SPEED_REGISTRY.show_all_results()

    ACC_REGISTRY.benchmark_all(init_params)
    ACC_REGISTRY.show_all_results()