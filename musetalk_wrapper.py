#!/usr/bin/env python3
"""
MuseTalk Wrapper Script - robust run with temp YAML, explicit PATH, rich logs
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path
from glob import glob

# MuseTalk install paths - use environment variables for container deployment
MUSETALK_PATH = os.environ.get("MUSETALK_PATH", "/app/MuseTalk")
FFMPEG_BIN   = os.environ.get("FFMPEG_BIN", "/usr/bin")
PYTHON_ENV   = os.environ.get("PYTHON_ENV", "python")

# Models (MuseTalk 1.5) - use forward slashes for Linux
UNET_PATH    = "models/musetalkV15/unet.pth"
UNET_CONFIG  = "models/musetalkV15/musetalk.json"
VERSION      = "v15"


def make_temp_yaml(audio_path: str, image_path: str, yaml_path: Path) -> None:
    """Create a valid MuseTalk normal-inference YAML with task_0 block."""
    img = image_path.replace("\\", "/")
    aud = audio_path.replace("\\", "/")

    # Minimal but valid structure matching MuseTalk's expected format
    yaml_text = (
        "task_0:\n"
        f"  video_path: \"{img}\"\n"
        f"  audio_path: \"{aud}\"\n"
        "  bbox_shift: 0\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

def check_required_weights() -> tuple[bool, str]:
    """Verify essential model files exist (e.g., sd-vae)."""
    sd_vae_bin = Path(MUSETALK_PATH) / "models" / "sd-vae" / "diffusion_pytorch_model.bin"
    sd_vae_safetensors = Path(MUSETALK_PATH) / "models" / "sd-vae" / "diffusion_pytorch_model.safetensors"
    if sd_vae_bin.exists() or sd_vae_safetensors.exists():
        return True, ""
    return False, str(sd_vae_bin)


def run_musetalk(audio_path: str, image_path: str, output_path: str) -> dict:
    try:
        # Ensure ffmpeg is visible to the subprocess
        os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

        # Work from MuseTalk directory
        os.chdir(MUSETALK_PATH)

        # Verify weights exist before long run
        ok, missing = check_required_weights()
        if not ok:
            return {"success": False, "error": f"Missing required weight file: {missing}"}

        # Use temp directory from environment variable or default
        temp_dir = os.environ.get("TEMP_DIR", "/tmp/results")
        result_dir = Path(temp_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary YAML (use consistent naming like our CMD tests)
        temp_yaml = result_dir / "my_test.yaml"
        make_temp_yaml(audio_path, image_path, temp_yaml)

        cmd = [
            PYTHON_ENV,
            "-m", "scripts.inference",
            "--inference_config", str(temp_yaml),
            "--result_dir", str(result_dir),
            "--unet_model_path", UNET_PATH,
            "--unet_config", UNET_CONFIG,
            "--version", VERSION,
            "--ffmpeg_path", FFMPEG_BIN,
            "--use_float16",
        ]

        # Ensure UTF-8 for stdout/stderr to avoid Windows charmap encode issues
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # up to 15 min for first run on 3050
            env=env,
        )

        # Persist raw logs to disk for offline inspection
        try:
            (result_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
            (result_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8", errors="ignore")
        except Exception:
            pass

        run_logs = {
            "cmd": cmd,
            "cwd": MUSETALK_PATH,
            "stdout": proc.stdout,            # full output for deep diagnostics
            "stderr": proc.stderr,
            "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
            "returncode": proc.returncode,
            "result_dir": str(result_dir),
        }

        if proc.returncode != 0:
            return {"success": False, "error": "MuseTalk returned non-zero", "logs": run_logs}

        # Find an MP4 in result_dir (MuseTalk may choose name/dir)
        mp4s = sorted(glob(str(result_dir / "**" / "*.mp4"), recursive=True))
        if not mp4s:
            # Try fallback search under MuseTalk default results directory
            fallback_root = Path(MUSETALK_PATH) / "results"
            fallback_mp4s = []
            if fallback_root.exists():
                fallback_mp4s = sorted(glob(str(fallback_root / "**" / "*.mp4"), recursive=True))

            # Extend logs with directory listings to show in Network console
            run_logs["result_dir_files"] = [str(p) for p in (result_dir.rglob("*")) if p.is_file()][:200]
            if fallback_root.exists():
                run_logs["fallback_results_root"] = str(fallback_root)
                run_logs["fallback_results_files"] = [str(p) for p in (fallback_root.rglob("*.mp4"))][:200]

            if fallback_mp4s:
                best = max(fallback_mp4s, key=lambda p: (Path(p).stat().st_size, Path(p).stat().st_mtime))
                # Copy/move the best fallback into our expected output
                Path(best).replace(output_path)
                return {"success": True, "output": output_path, "logs": run_logs}

            return {"success": False, "error": "No MP4 found in result_dir", "logs": run_logs}

        best = max(mp4s, key=lambda p: (Path(p).stat().st_size, Path(p).stat().st_mtime))

        # Move to expected output_path
        Path(best).replace(output_path)

        return {"success": True, "output": output_path, "logs": run_logs}

    except subprocess.TimeoutExpired as e:
        return {"success": False, "error": "MuseTalk timeout expired", "logs": {"phase": "timeout", "exception": repr(e)}}
    except Exception as e:
        # best-effort: include any partial stdout/stderr if available
        return {"success": False, "error": str(e), "logs": {"exception": repr(e)}}


def main():
    try:
        if len(sys.argv) != 4:
            print(json.dumps({"success": False, "error": "Usage: musetalk_wrapper.py <audio_path> <image_path> <output_path>"}))
            sys.exit(1)

        audio_path, image_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

        if not Path(audio_path).exists():
            print(json.dumps({"success": False, "error": f"Audio file not found: {audio_path}"})); sys.exit(1)
        if not Path(image_path).exists():
            print(json.dumps({"success": False, "error": f"Image file not found: {image_path}"})); sys.exit(1)

        result = run_musetalk(audio_path, image_path, output_path)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


if __name__ == "__main__":
    main()
