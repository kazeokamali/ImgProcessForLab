import os
import numpy as np
import tifffile as tiff
import imageio.v3 as iio
from typing import List, Callable, Optional
from .cuda_check import check_cuda, get_array_module, to_gpu, to_cpu, CUDA_AVAILABLE

check_cuda()
xp = get_array_module()


class GPUBatchOperations:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.progress_callback = None

    def set_progress_callback(self, callback: Callable[[int, int], None]):
        self.progress_callback = callback

    def _load_batch_to_gpu(self, image_paths: List[str]):
        gpu_images = []
        for path in image_paths:
            img = iio.imread(path)
            if img.ndim == 3:
                img = img[:, :, 0]
            gpu_images.append(to_gpu(img.astype(np.float32)))
        return gpu_images

    def _save_batch_to_disk(self, gpu_images, output_paths: List[str]):
        for gpu_img, output_path in zip(gpu_images, output_paths):
            cpu_img = to_cpu(gpu_img)
            tiff.imwrite(output_path, cpu_img.astype(np.float32), dtype='float32')

    def batch_negative_log(self, image_paths: List[str], output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        
        processed_count = 0
        total_images = len(image_paths)
        index = 0
        
        while index < total_images:
            batch_end = min(index + self.batch_size, total_images)
            batch_paths = image_paths[index:batch_end]
            
            try:
                gpu_images = self._load_batch_to_gpu(batch_paths)
                
                for img in gpu_images:
                    safe_img = xp.maximum(img, 1e-5)
                    img[:] = -xp.log(safe_img)
                
                output_paths = []
                for path in batch_paths:
                    output_path = os.path.join(output_folder, os.path.basename(path))
                    output_paths.append(output_path)
                
                self._save_batch_to_disk(gpu_images, output_paths)
                
                processed_count += len(gpu_images)
                index = batch_end
                
                if self.progress_callback:
                    self.progress_callback(processed_count, total_images)
                
                del gpu_images
                if CUDA_AVAILABLE:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                if CUDA_AVAILABLE and 'OutOfMemoryError' in str(type(e)):
                    if self.batch_size > 1:
                        self.batch_size = max(1, self.batch_size // 2)
                        continue
                    else:
                        raise
                else:
                    raise

    def batch_rotate_r90(self, image_paths: List[str], output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        
        processed_count = 0
        total_images = len(image_paths)
        index = 0
        
        while index < total_images:
            batch_end = min(index + self.batch_size, total_images)
            batch_paths = image_paths[index:batch_end]
            
            try:
                gpu_images = self._load_batch_to_gpu(batch_paths)
                
                for i, img in enumerate(gpu_images):
                    gpu_images[i] = xp.rot90(img, k=3)
                
                output_paths = []
                for path in batch_paths:
                    output_path = os.path.join(output_folder, os.path.basename(path))
                    output_paths.append(output_path)
                
                self._save_batch_to_disk(gpu_images, output_paths)
                
                processed_count += len(gpu_images)
                index = batch_end
                
                if self.progress_callback:
                    self.progress_callback(processed_count, total_images)
                
                del gpu_images
                if CUDA_AVAILABLE:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                if CUDA_AVAILABLE and 'OutOfMemoryError' in str(type(e)):
                    if self.batch_size > 1:
                        self.batch_size = max(1, self.batch_size // 2)
                        continue
                    else:
                        raise
                else:
                    raise
