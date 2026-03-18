import os
import numpy as np
import tifffile as tiff
from PIL import Image
from scipy import ndimage as np_ndimage
from .cuda_check import check_cuda, get_array_module, to_gpu, to_cpu, CUDA_AVAILABLE

check_cuda()
xp = get_array_module()
cp = xp

try:
    if CUDA_AVAILABLE:
        from cupyx.scipy import ndimage as xp_ndimage
    else:
        xp_ndimage = None
except Exception:
    xp_ndimage = None


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
    os.makedirs(output_path, exist_ok=True)
    raw_data = np.fromfile(input_path, dtype=np.uint16)
    expected_size = int(width) * int(height)
    if raw_data.size != expected_size:
        raise ValueError(
            f"RAW尺寸不匹配: 设置为 {width}x{height} (共{expected_size}像素), 实际 {raw_data.size}"
        )

    raw_reshaped = raw_data.reshape((int(height), int(width)))
    gpu_img = to_gpu(raw_reshaped)
    output_filename = os.path.join(output_path, os.path.basename(input_path).replace('.raw', '.tif'))
    tiff.imwrite(output_filename, to_cpu(gpu_img).astype(np.uint16, copy=False), dtype=np.uint16)
    return output_filename


def gpu_convert_bit_depth(img_path, output_path, target_bit_depth, raw_width=2340, raw_height=2882):
    """Convert image bit depth with GPU acceleration when CUDA is available.

    target_bit_depth: 8, 16, or 32 (32 maps to float32).
    """
    os.makedirs(output_path, exist_ok=True)

    ext = os.path.splitext(img_path)[1].lower()
    if ext == ".raw":
        raw_data = np.fromfile(img_path, dtype=np.uint16)
        expected_size = int(raw_width) * int(raw_height)
        if raw_data.size != expected_size:
            raise ValueError(
                f"RAW size mismatch: expected {raw_width}x{raw_height}={expected_size}, got {raw_data.size}"
            )
        src_img = raw_data.reshape((int(raw_height), int(raw_width)))
    elif ext in (".tif", ".tiff"):
        src_img = tiff.imread(img_path)
    else:
        src_img = np.array(Image.open(img_path))

    if src_img.ndim == 3 and src_img.shape[-1] == 1:
        src_img = src_img[..., 0]

    src_dtype = np.dtype(src_img.dtype)
    gpu_img = to_gpu(src_img)

    if int(target_bit_depth) == 32:
        converted_gpu = gpu_img.astype(cp.float32, copy=False)
        save_dtype = np.float32
    elif int(target_bit_depth) in (8, 16):
        img_f32 = gpu_img.astype(cp.float32, copy=False)
        if src_dtype.kind in ("u", "i"):
            src_info = np.iinfo(src_dtype)
            src_min = float(src_info.min)
            src_max = float(src_info.max)
            denom = max(src_max - src_min, 1.0)
            norm = (img_f32 - src_min) / denom
        else:
            finite_mask = cp.isfinite(img_f32)
            if bool(to_cpu(cp.any(finite_mask))):
                finite_vals = img_f32[finite_mask]
                src_min = cp.min(finite_vals)
                src_max = cp.max(finite_vals)
                denom = cp.maximum(src_max - src_min, cp.float32(1e-12))
                norm = (img_f32 - src_min) / denom
            else:
                norm = cp.zeros_like(img_f32, dtype=cp.float32)
            norm = cp.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)

        norm = cp.clip(norm, 0.0, 1.0)
        if int(target_bit_depth) == 8:
            converted_gpu = (norm * 255.0 + 0.5).astype(cp.uint8)
            save_dtype = np.uint8
        else:
            converted_gpu = (norm * 65535.0 + 0.5).astype(cp.uint16)
            save_dtype = np.uint16
    else:
        raise ValueError("target_bit_depth must be one of: 8, 16, 32")

    converted_cpu = to_cpu(converted_gpu)

    base_name, base_ext = os.path.splitext(os.path.basename(img_path))
    if base_ext.lower() not in (".tif", ".tiff"):
        base_ext = ".tif"
    if ext == ".raw":
        base_ext = ".tif"
    output_filename = os.path.join(output_path, f"{base_name}{base_ext}")

    tiff.imwrite(
        output_filename,
        converted_cpu.astype(save_dtype, copy=False),
        dtype=save_dtype,
    )
    return output_filename, str(src_dtype), str(np.dtype(save_dtype))


def gpu_crop_image(img_path, output_path, x1, y1, x2, y2):
    os.makedirs(output_path, exist_ok=True)
    img = tiff.imread(img_path)
    gpu_img = to_gpu(img)

    h, w = gpu_img.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"无效裁剪区域: ({x1}, {y1}) -> ({x2}, {y2})")

    cropped = gpu_img[y1:y2, x1:x2]
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    cropped_cpu = to_cpu(cropped)
    tiff.imwrite(output_filename, cropped_cpu.astype(cropped_cpu.dtype, copy=False), dtype=cropped_cpu.dtype)
    return output_filename


def gpu_rotate_image(img_path, output_path, angle):
    os.makedirs(output_path, exist_ok=True)
    img = tiff.imread(img_path).astype(np.float32, copy=False)

    if CUDA_AVAILABLE and xp_ndimage is not None:
        gpu_img = to_gpu(img)
        rotated_gpu = xp_ndimage.rotate(
            gpu_img,
            angle=float(angle),
            reshape=True,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        rotated_cpu = to_cpu(rotated_gpu)
    else:
        rotated_cpu = np_ndimage.rotate(
            img,
            angle=float(angle),
            reshape=True,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32, copy=False)

    output_filename = os.path.join(output_path, os.path.basename(img_path))
    tiff.imwrite(output_filename, rotated_cpu.astype(np.float32, copy=False), dtype=np.float32)
    return output_filename


def gpu_sharpen_edge(img_path, output_path):
    from scipy.ndimage import gaussian_filter
    
    os.makedirs(output_path, exist_ok=True)
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
    sharpened_cpu = to_cpu(sharpened_image)
    tiff.imwrite(output_filename, sharpened_cpu.astype(np.float32, copy=False), dtype=np.float32)
    return output_filename


def gpu_negative_log(img_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    img = np.array(Image.open(img_path))
    gpu_img = cp.asarray(img)
    gpu_img = cp.where(gpu_img <= 0, 1e-5, gpu_img)
    ln_img = -(cp.log(gpu_img))
    
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    ln_cpu = to_cpu(ln_img)
    tiff.imwrite(output_filename, ln_cpu.astype(np.float32, copy=False), dtype=np.float32)
    return output_filename


def gpu_rotate_r90(img_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    img = np.array(Image.open(img_path))
    gpu_img = cp.asarray(img)
    rotated = cp.rot90(gpu_img, k=3)
    
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    rotated_cpu = to_cpu(rotated)
    tiff.imwrite(output_filename, rotated_cpu.astype(rotated_cpu.dtype, copy=False), dtype=rotated_cpu.dtype)
    return output_filename
