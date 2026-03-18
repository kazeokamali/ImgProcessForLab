from typing import List

from src.lifton2019.io_loader import collect_image_files


def collect_projection_files(folder: str) -> List[str]:
    return collect_image_files(folder)
