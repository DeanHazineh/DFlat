import zipfile
import os
import shutil


DROPBOX_LINK = "https://www.dropbox.com/sh/jyubstf963x34jl/AADpVbN3dTSxhVOo3Hy1m2_8a?dl=1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_PATH = os.path.join(SCRIPT_DIR, "all_data.zip")
UNZIP_PATH = os.path.join(SCRIPT_DIR, "temp_fold/")


def download_file(url, save_path):
    import requests

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return


def unzip_file(zip_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # Unzip the main dropbox folder
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return


def unpack_and_move():
    zip_file = ["pregen_lookup_tables.zip", "raw_meta_libraries.zip", "validation_scripts.zip", "trained_MLP_models.zip"]
    dflat_loc = [
        "dflat/metasurface_library/core/",
        "dflat/metasurface_library/core/",
        "dflat/fourier_layer/core/",
        "dflat/neural_optical_layer/core/",
        "dflat/physical_optical_layer/core/",
    ]

    for idx, zfile in enumerate(zip_file):
        zfile_loc = os.path.join(UNZIP_PATH, zfile)
        store_at = os.path.join(SCRIPT_DIR, dflat_loc[idx])
        unzip_file(zfile_loc, store_at)

    return


def execute_data_management():
    print("Downloading Dflat data from dropbox...")
    download_file(DROPBOX_LINK, DEST_PATH)

    print("Unzipping data...")
    unzip_file(DEST_PATH, UNZIP_PATH)

    print("Moving data files to dflat folders")
    unpack_and_move()

    print("Cleaning and deleting the initial zip file")
    os.remove(DEST_PATH)
    shutil.rmtree(UNZIP_PATH)

    print("Data retrieving finished")

    return
