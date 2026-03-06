import os
import shutil
import multiprocessing
from tqdm import tqdm


def copy_file(source_file, dest_file):
    if os.path.exists(dest_file):
        return
    try:
        shutil.copy(source_file, dest_file)
        if dest_file.endswith("01_0000.gz"):
            print(dest_file)
    except Exception as e:
        print(e, os.path.basename(source_file))


def transfer_list(data_directory, dst, yrmo_str=None, workers=2):
    files_ = sorted(os.listdir(data_directory))
    yrmo = [str(f[:6]) for f in files_]

    if yrmo_str:
        file_list = [
            os.path.join(data_directory, f)
            for f, ym in zip(files_, yrmo)
            if ym in yrmo_str
        ]
        dst_list = [
            os.path.join(dst, f) for f, ym in zip(files_, yrmo) if ym in yrmo_str
        ]
    else:
        file_list = [os.path.join(data_directory, f) for f in files_]
        dst_list = [os.path.join(dst, f) for f in files_]

    print(f"{len(file_list)} files to transfer.")

    with multiprocessing.Pool(processes=workers) as pool:
        tqdm(
            pool.starmap(copy_file, zip(file_list, dst_list)),
            total=len(file_list),
            desc="Transferring files",
            unit="file",
        )


if __name__ == "__main__":
    pass
# ========================= EOF ====================================================================
