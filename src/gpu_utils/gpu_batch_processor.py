import os
import numpy as np
import imageio.v3 as iio
import tifffile as tiff
from typing import List, Tuple, Callable, Optional
from .cuda_check import check_cuda, get_array_module, to_gpu, to_cpu, CUDA_AVAILABLE
from .gpu_memory_manager import GPUMemoryManager
import cupy as cp



check_cuda()
xp = get_array_module()


class GPUBatchProcessor:
    def __init__(self, memory_usage_percent=0.95, num_streams=2, batch_size=100):
        self.memory_manager = GPUMemoryManager(memory_usage_percent) if CUDA_AVAILABLE else None
        self.num_streams = num_streams
        self.streams = []
        if CUDA_AVAILABLE:
            import cupy as cp
            self.streams = [cp.cuda.Stream() for _ in range(num_streams)]
        self.batch_size = batch_size
        self.progress_callback = None

    def set_progress_callback(self, callback: Callable[[int, int, dict], None]):
        self.progress_callback = callback

    def _load_batch_to_gpu(self, image_paths: List[str], stream_idx: int = 0):
        gpu_images = []
        
        for path in image_paths:
            img = iio.imread(path)
            if img.ndim == 3:
                img = img[:, :, 0]
            
            if CUDA_AVAILABLE and self.streams:
                with self.streams[stream_idx]:
                    gpu_img = to_gpu(img.astype(np.float32))
                    gpu_images.append(gpu_img)
            else:
                gpu_img = to_gpu(img.astype(np.float32))
                gpu_images.append(gpu_img)
        
        return gpu_images

    def _save_batch_to_disk(self, gpu_images, output_paths: List[str], stream_idx: int = 0):
        for gpu_img, output_path in zip(gpu_images, output_paths):
            if CUDA_AVAILABLE and self.streams:
                with self.streams[stream_idx]:
                    cpu_img = to_cpu(gpu_img)
                    tiff.imwrite(output_path, cpu_img.astype(np.float32), dtype='float32')
            else:
                cpu_img = to_cpu(gpu_img)
                tiff.imwrite(output_path, cpu_img.astype(np.float32), dtype='float32')

    def batch_subtract(self, images, subtract_img):
        subtract_float = subtract_img.astype(np.float32)
        for img in images:
            img -= subtract_float
        return images

    def batch_divide(self, images, divide_img):
        divide_float = divide_img.astype(np.float32)
        for img in images:
            safe_divide = xp.where(divide_float != 0, divide_float, 1.0)
            img[:] = xp.divide(img, safe_divide)
        return images

    def batch_delete_blacklines_inRange(
        self,
        images,
        min_val: float,
        max_val: float,
        blackline_lp: float,
        if_process_all: bool = False,
        blacklines_columns: Optional[List[int]] = None
    ):
        height, width = images[0].shape
        
        for img in images:
            if not if_process_all:
                for col in blacklines_columns:
                    img[:, col] = (img[:, col - 1] + img[:, col + 1]) / 2.0
            else:
                mask = (img >= min_val) & (img <= max_val)
                invalid_value_nums = xp.sum(~mask, axis=0).astype(np.int32)
                
                blackline_length = height / blackline_lp
                bad_cols = xp.where(invalid_value_nums >= blackline_length)[0]
                
                for col in bad_cols:
                    if 1 < col < width - 2:
                        img[1:-1, col] = (img[1:-1, col - 1] + img[1:-1, col + 1]) / 2.0
        
        return images

    def batch_delete_blacklines_inGrad(
        self,
        images,
        grad: float,
        blackline_lp: float,
        if_process_all: bool = False,
        blacklines_columns: Optional[List[int]] = None
    ):
        height, width = images[0].shape
        
        for img in images:
            left_col = xp.roll(img, 1, axis=1)
            right_col = xp.roll(img, -1, axis=1)
            
            p_bar = (left_col + right_col) / 2.0
            pixel = xp.where(img > 0, img, 1e-5)
            p_bar_safe = xp.where(p_bar > 0, p_bar, 1e-5)
            
            grad1 = xp.abs((p_bar - pixel) / (pixel + 1e-5))
            grad2 = xp.abs((p_bar - pixel) / (p_bar_safe + 1e-5))
            
            valid_grad_mask = (grad1 <= grad) & (grad2 <= grad)
            
            invalid_value_nums = xp.sum(~valid_grad_mask, axis=0).astype(np.int32)
            
            blackline_length = height / blackline_lp
            
            if not if_process_all:
                for col in blacklines_columns:
                    if 0 < col < width - 1:
                        img[:, col] = (img[:, col - 1] + img[:, col + 1]) / 2.0
            else:
                bad_cols = xp.where(invalid_value_nums >= blackline_length)[0]
                for col in bad_cols:
                    if 0 < col < width - 1:
                        img[:, col] = (img[:, col - 1] + img[:, col + 1]) / 2.0
        
        return images

    def batch_negative_log(self, images):
        for img in images:
            safe_img = xp.maximum(img, 1e-5)
            img[:] = -xp.log(safe_img)
        return images

    def batch_rotate_r90(self, images):
        for i, img in enumerate(images):
            images[i] = xp.rot90(img, k=3)
        return images

    def process_tomos_batch(
        self,
        tomo_paths: List[str],
        black_img_path: str,
        white_sub_black_img_path: str,
        output_folder: str,
        min_val: float,
        max_val: float,
        blackline_lp: float,
        if_process_all: bool = False,
        blacklines_columns: Optional[List[int]] = None
    ):
        os.makedirs(output_folder, exist_ok=True)
        
        black_img_cpu = iio.imread(black_img_path)
        if black_img_cpu.ndim == 3:
            black_img_cpu = black_img_cpu[:, :, 0]
        black_img_gpu = cp.asarray(black_img_cpu, dtype=cp.float32)
        
        white_sub_black_cpu = iio.imread(white_sub_black_img_path)
        if white_sub_black_cpu.ndim == 3:
            white_sub_black_cpu = white_sub_black_cpu[:, :, 0]
        white_sub_black_gpu = cp.asarray(white_sub_black_cpu, dtype=cp.float32)
        
        processed_count = 0
        total_images = len(tomo_paths)
        index = 0
        
        while index < total_images:
            batch_end = min(index + self.batch_size, total_images)
            batch_paths = tomo_paths[index:batch_end]
            
            try:
                gpu_images = self._load_batch_to_gpu(batch_paths)
                
                self.batch_subtract(gpu_images, black_img_gpu)
                self.batch_delete_blacklines_inRange(
                    gpu_images, min_val, max_val, blackline_lp, if_process_all, blacklines_columns
                )
                
                output_paths = []
                for path in batch_paths:
                    basename = os.path.basename(path)
                    if basename.endswith('.tiff'):
                        output_path = os.path.join(output_folder, basename.replace('.tiff', '_.tiff'))
                    elif basename.endswith('.tif'):
                        output_path = os.path.join(output_folder, basename.replace('.tif', '_.tif'))
                    else:
                        output_path = os.path.join(output_folder, basename)
                    output_paths.append(output_path)
                
                self._save_batch_to_disk(gpu_images, output_paths)
                
                processed_count += len(gpu_images)
                index = batch_end
                
                if self.progress_callback:
                    self.progress_callback(processed_count, total_images, self.memory_manager.get_memory_info())
                
                for stream in self.streams:
                    stream.synchronize()
                
                del gpu_images
                cp.get_default_memory_pool().free_all_blocks()
                
            except cp.cuda.memory.OutOfMemoryError:
                self.text.append(f'<span style="color: red;">  警告: 显存不足，批次大小 {len(batch_paths)} 太大，尝试减小...</span>')
                if self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    self.text.append(f'<span style="color: orange;">  自动调整批次大小为: {self.batch_size}</span>')
                    continue
                else:
                    self.text.append(f'<span style="color: red;">  错误: 批次大小已为1，仍然显存不足！</span>')
                    raise
        
        del black_img_gpu
        del white_sub_black_gpu
        cp.get_default_memory_pool().free_all_blocks()

    def process_divide_batch(
        self,
        input_paths: List[str],
        divide_img_path: str,
        output_folder: str,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None
    ):
        os.makedirs(output_folder, exist_ok=True)
        
        divide_img_cpu = iio.imread(divide_img_path)
        if divide_img_cpu.ndim == 3:
            divide_img_cpu = divide_img_cpu[:, :, 0]
        divide_img_gpu = cp.asarray(divide_img_cpu, dtype=cp.float32)
        
        processed_count = 0
        total_images = len(input_paths)
        index = 0
        
        while index < total_images:
            batch_end = min(index + self.batch_size, total_images)
            batch_paths = input_paths[index:batch_end]
            
            try:
                gpu_images = self._load_batch_to_gpu(batch_paths)
                
                self.batch_divide(gpu_images, divide_img_gpu)
                
                output_paths = []
                for path in batch_paths:
                    output_path = os.path.join(output_folder, os.path.basename(path))
                    output_paths.append(output_path)
                
                self._save_batch_to_disk(gpu_images, output_paths)
                
                processed_count += len(gpu_images)
                index = batch_end
                
                if progress_callback:
                    progress_callback(processed_count, total_images, self.memory_manager.get_memory_info())
                
                for stream in self.streams:
                    stream.synchronize()
                
                del gpu_images
                cp.get_default_memory_pool().free_all_blocks()
                
            except cp.cuda.memory.OutOfMemoryError:
                if self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    continue
                else:
                    raise
        
        del divide_img_gpu
        cp.get_default_memory_pool().free_all_blocks()

    def get_memory_info(self) -> dict:
        return self.memory_manager.get_memory_info()
