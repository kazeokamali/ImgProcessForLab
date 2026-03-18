import argparse
import json
import sys
import traceback
from dataclasses import asdict
from typing import List

from .fdk_runner import run_fdk_reconstruction
from .models import ReconstructionConfig


def _sanitize_text(text: str) -> str:
    return str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _emit(tag: str, *parts):
    payload = "\t".join([tag] + [_sanitize_text(p) for p in parts])
    print(payload, flush=True)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruction worker subprocess entry.")
    parser.add_argument("--config-json", required=True, help="Path to serialized ReconstructionConfig JSON.")
    parser.add_argument(
        "--projection-list-json",
        required=True,
        help="Path to projection files JSON list.",
    )
    parser.add_argument("--result-json", required=True, help="Path to write worker result JSON.")
    return parser


def _progress(done: int, total: int, message: str):
    _emit("PROGRESS", done, total, message)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        config_payload = _load_json(args.config_json)
        projection_files = _load_json(args.projection_list_json)

        # Backward compatibility for older stage config files.
        config_payload.setdefault("iterative_iterations", 0)

        if not isinstance(projection_files, list):
            raise ValueError("projection_files JSON 必须是字符串数组。")
        projection_files = [str(x) for x in projection_files]

        config = ReconstructionConfig(**config_payload)
        _emit("INFO", "worker_started")

        result = run_fdk_reconstruction(
            config=config,
            projection_files=projection_files,
            progress_callback=_progress,
            stop_requested=None,
        )

        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        _emit("RESULT", args.result_json)
        return 0
    except Exception as e:
        _emit("ERROR", str(e))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
