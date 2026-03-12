import cupy as cp
from typing import List, Tuple


class GPUMemoryManager:
    def __init__(self, memory_usage_percent=0.95):
        self.memory_usage_percent = memory_usage_percent
        self.total_memory = self._get_total_gpu_memory()
        self.usable_memory = int(self.total_memory * memory_usage_percent)
        self.allocated_memory = 0
        self.memory_pool = cp.get_default_memory_pool()
        self.memory_pool.set_limit(size=self.usable_memory)

    def _get_total_gpu_memory(self) -> int:
        mem = cp.cuda.Device().mem_info
        return mem[1]

    def get_total_memory(self) -> int:
        return self.total_memory

    def get_usable_memory(self) -> int:
        return self.usable_memory

    def get_free_memory(self) -> int:
        mem = cp.cuda.Device().mem_info
        return mem[0]

    def get_used_memory(self) -> int:
        return self.allocated_memory

    def can_allocate(self, size: int) -> bool:
        return (self.get_free_memory() >= size) and (self.allocated_memory + size <= self.usable_memory)

    def get_available_batch_size(self, image_size: int, min_batch: int = 1) -> int:
        free = min(self.get_free_memory(), self.usable_memory - self.allocated_memory)
        max_possible = free // image_size
        return max(min_batch, max_possible)

    def track_allocation(self, size: int):
        self.allocated_memory += size

    def track_free(self, size: int):
        self.allocated_memory = max(0, self.allocated_memory - size)

    def get_memory_info(self) -> dict:
        mem = cp.cuda.Device().mem_info
        return {
            "total": self.total_memory,
            "usable": self.usable_memory,
            "free": mem[0],
            "used": self.allocated_memory,
            "free_mb": mem[0] / (1024 ** 2),
            "total_mb": self.total_memory / (1024 ** 2),
            "used_mb": self.allocated_memory / (1024 ** 2),
            "usage_percent": (self.allocated_memory / self.usable_memory) * 100
        }

    def reset(self):
        self.allocated_memory = 0
        self.memory_pool.free_all_blocks()
