import os
import numpy as np
import tifffile as tiff
from PIL import Image
from .cuda_check import check_cuda, get_array_module, to_gpu, to_cpu, CUDA_AVAILABLE
import cupy as cp   

check_cuda()
xp = get_array_module()


class GPUImageProcess:
    def __init__(self, img_path):
        self.width = 0
        self.height = 0
        self.img_path = img_path
        self.img = self._load_image_gpu(img_path)
        self.blackline_length = 0.0
        self.blackline_lp = 2
        self.max_val = 8000
        self.min_val = 2000
        self.valid_grad = 4.0
        self.blacklines_columns = None
        self.If_Process_AllColumns = False

    def _load_image_gpu(self, img_path):
        if img_path.endswith('.raw'):
            raw_data = np.fromfile(img_path, dtype=np.uint16)
            height, width = 2882, 2340
            self.height, self.width = height, width
            return to_gpu(raw_data.reshape((height, width)))
        else:
            img = np.array(Image.open(img_path))
            self.height, self.width = img.shape[0], img.shape[1]
            return to_gpu(img)

    def get_length(self):
        self.height, self.width = float(self.img.shape[0]), float(self.img.shape[1])

    def get_pixel_value(self, i, j):
        return self.img[j, i]

    def set_pixel_value(self, i, j, val):
        self.img[j, i] = val

    def is_pixel_valid(self, i, j):
        if self.max_val >= self.img[j, i] >= self.min_val:
            return True
        return False

    def is_grad_permitted(self, i, j):
        p_bar = (self.get_pixel_value(i + 1, j) + self.get_pixel_value(i - 1, j)) / 2
        pixel = self.get_pixel_value(i, j) if self.get_pixel_value(i, j) > 0 else 1e-5
        if abs((p_bar - pixel) / (pixel + 1e-5)) <= self.valid_grad:
            if abs((p_bar - pixel) / (p_bar + 1e-5)) <= self.valid_grad:
                return True
        return False

    def save_to_path(self, save_path):
        img_cpu = to_cpu(self.img)
        tiff.imwrite(save_path, img_cpu.astype(np.float32), dtype='float32')

    def delete_blacklines_inRange(self):
        self.get_length()
        if not self.If_Process_AllColumns:
            Columns_BlackLines = self.blacklines_columns
            for i in Columns_BlackLines:
                for j in range(int(self.height)):
                    temp = float(
                        self.get_pixel_value(i - 1, j) + self.get_pixel_value(i + 1, j)
                    )
                    temp = temp / 2.0
                    self.set_pixel_value(i, j, temp)
        else:
            Columns_BlackLines = list(range(0, int(self.width)))
            invalid_value_nums = cp.zeros(int(self.width), dtype=cp.int32)
            
            for i in Columns_BlackLines:
                for j in range(int(self.height)):
                    if not self.is_pixel_valid(i, j):
                        invalid_value_nums[i] += 1

            self.blackline_length = self.height / self.blackline_lp
            for i in Columns_BlackLines[1:-2]:
                if invalid_value_nums[i] >= self.blackline_length:
                    for j in range(1, int(self.height) - 1):
                        temp = float(
                            self.get_pixel_value(i - 1, j)
                            + self.get_pixel_value(i + 1, j)
                        )
                        temp = temp / 2.0
                        self.set_pixel_value(i, j, temp)

    def delete_blacklines_inGrad(self):
        self.get_length()
        if not self.If_Process_AllColumns:
            Columns_BlackLines = self.blacklines_columns
        else:
            Columns_BlackLines = list(range(0, int(self.width)))

        invalid_value_nums = cp.zeros(int(self.width), dtype=cp.int32)

        for i in Columns_BlackLines:
            for j in range(int(self.height)):
                if not self.is_grad_permitted(i, j):
                    invalid_value_nums[i] += 1

        self.blackline_length = self.height / self.blackline_lp

        for i in Columns_BlackLines:
            if invalid_value_nums[i] >= self.blackline_length:
                for j in range(int(self.height)):
                    temp = float(
                        self.get_pixel_value(i - 1, j) + self.get_pixel_value(i + 1, j)
                    )
                    temp = temp / 2.0
                    self.set_pixel_value(i, j, temp)

    def image_subtract(self, other):
        self.get_length()
        other.get_length()
        self_img = self.img.astype(cp.float32)
        other_img = other.img.astype(cp.float32)
        self.img = self_img - other_img

    def image_divide(self, other):
        try:
            self.get_length()
            other.get_length()
            self_img = self.img.astype(cp.float32)
            other_img = other.img.astype(cp.float32)
            result = cp.where(other_img != 0, self_img / other_img, 0)
            self.img = result
        except Exception as e:
            print(f"Error in image_divide: {e}")

    def image_add(self, other):
        self.get_length()
        other.get_length()
        self_img = self.img.astype(cp.float32)
        other_img = other.img.astype(cp.float32)
        self.img = self_img + other_img


def gpu_convert_raw_to_tiff(input_path, output_path, width=2340, height=2882):
    raw_data = np.fromfile(input_path, dtype=np.uint16)
    raw_reshaped = raw_data.reshape((height, width))
    gpu_img = cp.asarray(raw_reshaped)
    output_filename = os.path.join(output_path, os.path.basename(input_path).replace('.raw', '.tif'))
    tiff.imwrite(output_filename, cp.asnumpy(gpu_img), dtype='uint16')
    return output_filename


def gpu_crop_image(img_path, output_path, x1, y1, x2, y2):
    img = np.array(Image.open(img_path))
    gpu_img = cp.asarray(img)
    cropped = gpu_img[y1:y2, x1:x2]
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    Image.fromarray(cp.asnumpy(cropped)).save(output_filename)
    return output_filename


def gpu_rotate_image(img_path, output_path, angle):
    img = np.array(Image.open(img_path))
    gpu_img = cp.asarray(img)
    
    angle_rad = cp.deg2rad(angle)
    cos_a = cp.cos(angle_rad)
    sin_a = cp.sin(angle_rad)
    
    height, width = gpu_img.shape[:2]
    new_width = int(cp.abs(width * cos_a) + cp.abs(height * sin_a))
    new_height = int(cp.abs(width * sin_a) + cp.abs(height * cos_a))
    
    if gpu_img.ndim == 2:
        rotated_gpu = cp.zeros((new_height, new_width), dtype=gpu_img.dtype)
    else:
        rotated_gpu = cp.zeros((new_height, new_width, gpu_img.shape[2]), dtype=gpu_img.dtype)
    
    for y in range(new_height):
        for x in range(new_width):
            x_centered = x - new_width / 2
            y_centered = y - new_height / 2
            
            x_original = x_centered * cos_a - y_centered * sin_a + width / 2
            y_original = x_centered * sin_a + y_centered * cos_a + height / 2
            
            if 0 <= x_original < width - 1 and 0 <= y_original < height - 1:
                x0 = int(cp.floor(x_original))
                y0 = int(cp.floor(y_original))
                x1 = x0 + 1
                y1 = y0 + 1
                
                dx = x_original - x0
                dy = y_original - y0
                
                if gpu_img.ndim == 2:
                    interpolated = (1 - dx) * (1 - dy) * gpu_img[y0, x0] + \
                                   dx * (1 - dy) * gpu_img[y0, x1] + \
                                   (1 - dx) * dy * gpu_img[y1, x0] + \
                                   dx * dy * gpu_img[y1, x1]
                    rotated_gpu[y, x] = interpolated
                else:
                    for c in range(gpu_img.shape[2]):
                        interpolated = (1 - dx) * (1 - dy) * gpu_img[y0, x0, c] + \
                                       dx * (1 - dy) * gpu_img[y0, x1, c] + \
                                       (1 - dx) * dy * gpu_img[y1, x0, c] + \
                                       dx * dy * gpu_img[y1, x1, c]
                        rotated_gpu[y, x, c] = interpolated
    
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    Image.fromarray(cp.asnumpy(rotated_gpu)).save(output_filename)
    return output_filename


def gpu_sharpen_edge(img_path, output_path):
    from scipy.ndimage import gaussian_filter
    
    image = np.array(Image.open(img_path))
    kernel = cp.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=cp.float32)
    
    if image.ndim == 3:
        sharpened_image = cp.zeros_like(image)
        for i in range(image.shape[2]):
            gpu_channel = cp.asarray(image[:, :, i])
            blurred = cp.asarray(gaussian_filter(image[:, :, i], sigma=1))
            blurred_gpu = cp.asarray(blurred)
            
            sharpened = cp.zeros_like(gpu_channel)
            for y in range(1, image.shape[0] - 1):
                for x in range(1, image.shape[1] - 1):
                    region = blurred_gpu[y-1:y+2, x-1:x+2]
                    sharpened[y, x] = cp.sum(region * kernel)
            
            sharpened_image[:, :, i] = sharpened
    else:
        gpu_img = cp.asarray(image)
        blurred = cp.asarray(gaussian_filter(image, sigma=1))
        blurred_gpu = cp.asarray(blurred)
        
        sharpened_image = cp.zeros_like(gpu_img)
        for y in range(1, image.shape[0] - 1):
            for x in range(1, image.shape[1] - 1):
                region = blurred_gpu[y-1:y+2, x-1:x+2]
                sharpened_image[y, x] = cp.sum(region * kernel)
    
    output_filename = os.path.join(output_path, f"sharpened_{os.path.basename(img_path)}")
    Image.fromarray(cp.asnumpy(sharpened_image)).save(output_filename)
    return output_filename


def gpu_negative_log(img_path, output_path):
    img = np.array(Image.open(img_path))
    gpu_img = cp.asarray(img)
    gpu_img = cp.where(gpu_img <= 0, 1e-5, gpu_img)
    ln_img = -(cp.log(gpu_img))
    
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    Image.fromarray(cp.asnumpy(ln_img).astype('float32')).save(output_filename)
    return output_filename


def gpu_rotate_r90(img_path, output_path):
    img = np.array(Image.open(img_path))
    gpu_img = cp.asarray(img)
    rotated = cp.rot90(gpu_img, k=3)
    
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    Image.fromarray(cp.asnumpy(rotated)).save(output_filename)
    return output_filename
