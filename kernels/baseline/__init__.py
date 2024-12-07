import torch
from utils.registry_factory import BENCH_REGISTRY


def init_torch_linear(A_shape, B_shape, A_data, B_data):
    assert A_shape == A_data.shape
    assert B_shape == B_data.shape
    torch_linear = (
        torch.nn.Linear(B_shape[0], B_shape[1], bias=False).cuda().to(B_data.dtype)
    )
    torch_linear.weight.data = B_data.t()
    return {"A_data": A_data, "torch_linear": torch_linear}


def run_torch_linear(A_data, torch_linear):
    torch_linear(A_data)


BENCH_REGISTRY.register("torch_linear", run_torch_linear, init_torch_linear)


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
        "torch_linear": {},
    }

    # BENCH_REGISTRY.benchmark("torch_linear", init_params)
    BENCH_REGISTRY.benchmark_all(init_params)

    BENCH_REGISTRY.show_all_results()
