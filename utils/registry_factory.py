import time
import gc
import torch
from tabulate import tabulate
from collections import defaultdict
import torch.nn.functional as F
from loguru import logger


class BenchRegister(dict):
    def __init__(self, *args, **kwargs):
        super(BenchRegister, self).__init__(*args, **kwargs)
        self._dict = {}
        self._results = defaultdict(dict)
        self._headers = ["Kernel", "Latency/s"]
        self.warmup_iter = kwargs.get("warmup_iter", 10)
        self.bench_iter = kwargs.get("bench_iter", 100)

    def register(self, name, kernel_func, init_func):
        self._dict[name] = {"kernel_func": kernel_func, "init_func": init_func}

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def sort_tabulate(self, data):
        for i, sublist in enumerate(data):
            if sublist[0] == "torch_matmul":
                k_sublist = data.pop(i)
                data.insert(0, k_sublist)

    def add_speedup(self, data):
        if len(data) < 2:
            return False
        baseline_latency = data[0][1]
        if baseline_latency == "-":
            return False
        data[0].append(1.00)
        for row in data[1:]:
            latency = row[1]
            if latency == "-":
                continue
            speedup = f"{baseline_latency / latency:.2f}"
            row.append(speedup)
        return True

    def show_all_results(self):
        data = []
        for name in self._results:
            data.append(
                [name] + [self._results[name].get(h, "-") for h in self._headers[1:]]
            )
        self.sort_tabulate(data)
        if self.add_speedup(data):
            headers = self._headers + ["Speed-Up"]
        else:
            headers = self._headers
        print(tabulate(data, headers=headers, tablefmt="psql", stralign="center"))

    def show_result(self, name):
        print(self._results[name])

    def check_run(self, DEVICE_FORBIDDEN, name, tag):
        cur_dev = torch.cuda.get_device_name(0)
        if name in DEVICE_FORBIDDEN:
            if isinstance(DEVICE_FORBIDDEN[name], list):
                for dev_forb in DEVICE_FORBIDDEN[name]:
                    if dev_forb in cur_dev:
                        logger.error(
                            f"Kernel {name} can not run on {cur_dev}. Skip it."
                        )
                        return False
            elif isinstance(DEVICE_FORBIDDEN[name], dict):
                if tag in DEVICE_FORBIDDEN[name]:
                    for dev_forb in DEVICE_FORBIDDEN[name][tag]:
                        if dev_forb in cur_dev:
                            logger.error(
                                f"Kernel {name} {tag} can not run on {cur_dev}. Skip it."
                            )
                            return False
        return True

    @torch.no_grad()
    def benchmark_func(self, name, kernel_init_params, kernel_tag):
        kernel = self[name]
        prepare_params = kernel["init_func"](**kernel_init_params)

        kernel_func = kernel["kernel_func"]
        for _ in range(self.warmup_iter):
            kernel_func(**prepare_params)

        torch.cuda.synchronize()
        tick = time.time()
        for _ in range(self.bench_iter):
            kernel_func(**prepare_params)
        torch.cuda.synchronize()
        result = (time.time() - tick) / self.bench_iter
        self._results[kernel_tag] = {"Latency/s": result}

        del prepare_params
        gc.collect()
        torch.cuda.empty_cache()

        time.sleep(1.0)

    @torch.no_grad()
    def benchmark(self, name, init_params):
        if not isinstance(init_params[name], list):
            init_params[name] = [{None: init_params[name]}]
        for init_param in init_params[name]:
            tag = next(iter(init_param))
            if init_param[tag] is None:
                init_param[tag] = {}
            kernel_init_params = init_params["default"] | init_param[tag]
            kernel_tag = name + " + " + tag if tag else name
            if self.check_run(init_params["DEVICE_FORBIDDEN"], name, tag):
                self.benchmark_func(name, kernel_init_params, kernel_tag)

    @torch.no_grad()
    def benchmark_all(self, init_params):
        for name in self.keys():
            if name in init_params:
                self.benchmark(name, init_params.copy())
                # try:
                #     self.benchmark(name, init_params.copy())
                # except Exception as e:
                #     logger.error(f"Kernel {name} has an error: {e}. Skip it.")


class SpeedRegister(BenchRegister):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AccRegister(BenchRegister):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_name = "torch_matmul"

    def compare_outputs(self, res):
        diff = torch.abs(self.baseline_res - res)
        max_diff = torch.max(diff)

        flatten_res = res.flatten()
        flatten_base_res = self.baseline_res.flatten()

        cosine_sim = F.cosine_similarity(
            flatten_res.to(torch.float64).unsqueeze(0),
            flatten_base_res.to(torch.float64).unsqueeze(0),
        ).item()
        return max_diff, cosine_sim

    @torch.no_grad()
    def benchmark_func(self, name, kernel_init_params, kernel_tag):
        kernel = self[name]
        prepare_params = kernel["init_func"](**kernel_init_params)
        kernel_func = kernel["kernel_func"]
        if name == self.baseline_name:
            self.baseline_res = kernel_func(**prepare_params)
            self._results[kernel_tag] = {"cosine": 1.0, "max_diff": 0.0}
        else:
            res = kernel_func(**prepare_params)
            max_diff, cosine_sim = self.compare_outputs(res)
            self._results[kernel_tag] = {"cosine": cosine_sim, "max_diff": max_diff}
        gc.collect()
        torch.cuda.empty_cache()

    def show_all_results(self):
        table_data = []
        for kernel, metrics in self._results.items():
            cosine = (
                f"{metrics['cosine']:.6e}"
                if not isinstance(metrics["cosine"], float)
                or not str(metrics["cosine"]).lower() == "nan"
                else "nan"
            )
            max_diff = (
                f"{metrics['max_diff']:.2f}"
                if isinstance(metrics["max_diff"], float)
                else metrics["max_diff"]
            )
            table_data.append([kernel, max_diff, cosine])
        headers = ["Kernel", "max_diff", "cosine"]
        print(tabulate(table_data, headers=headers, tablefmt="psql", stralign="center"))


SPEED_REGISTRY = SpeedRegister()
ACC_REGISTRY = AccRegister()
