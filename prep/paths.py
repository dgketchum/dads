import os
import shutil

DADS_ROOT = os.environ.get("DADS_DATA_ROOT", "/nas/dads")
RUN_ARCHIVE_ROOT = os.path.join(DADS_ROOT, "run_archive")
RUNS_ROOT = os.path.join(DADS_ROOT, "runs")

# Legacy compatibility alias. Historically many scripts and configs refer to
# `/nas/dads/mvp`, which is now a symlink to `/nas/dads/run_archive`.
MVP_ROOT = os.path.join(DADS_ROOT, "mvp")

# Local NVMe cache for heavy artifacts that are expensive to read from NAS.
NVME_CACHE = os.environ.get("DADS_NVME_CACHE", "/data/nvme0/dads_cache")


def stage_to_nvme(nas_path: str, subdir: str = "stage_b") -> str:
    """Copy a NAS file to local NVMe cache if stale or missing.

    Returns the local path. Falls back to *nas_path* if the NVMe
    volume is not writable.
    """
    local_dir = os.path.join(NVME_CACHE, subdir)
    local_path = os.path.join(local_dir, os.path.basename(nas_path))

    if not os.path.isdir(os.path.dirname(NVME_CACHE)):
        print(f"[stage_to_nvme] NVMe root not available, using NAS: {nas_path}")
        return nas_path

    os.makedirs(local_dir, exist_ok=True)

    needs_copy = True
    if os.path.exists(local_path):
        local_size = os.path.getsize(local_path)
        remote_size = os.path.getsize(nas_path)
        if local_size == remote_size:
            needs_copy = False

    if needs_copy:
        print(f"[stage_to_nvme] copying {nas_path} -> {local_path}")
        shutil.copy2(nas_path, local_path)
        print(f"[stage_to_nvme] done ({os.path.getsize(local_path) / 1e9:.2f} GB)")
    else:
        print(f"[stage_to_nvme] cache hit: {local_path}")

    return local_path
