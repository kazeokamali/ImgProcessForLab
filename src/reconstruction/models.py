from dataclasses import dataclass


@dataclass
class ReconstructionConfig:
    projection_folder: str
    output_folder: str
    projection_count: int

    sod_mm: float
    sdd_mm: float
    angle_step_deg: float
    start_angle_deg: float

    detector_pixel_size_x_mm: float
    detector_pixel_size_y_mm: float
    cor_offset_px: float

    algorithm: str
    iterative_iterations: int
    filter_name: str

    recon_nx: int
    recon_ny: int
    recon_nz: int

    output_format: str
    refine_iterations: int
    refine_step: float


@dataclass
class ReconstructionDerived:
    magnification: float
    voxel_size_x_mm: float
    voxel_size_y_mm: float
    voxel_size_z_mm: float
    total_scan_angle_deg: float
