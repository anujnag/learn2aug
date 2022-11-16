from google_drive_downloader import GoogleDriveDownloader as gdd

import numpy as np
import os
import torch


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Download Omniglot Dataset
    if not os.path.isdir("./omniglot_resized"):
        gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized.zip",
            unzip=True,
        )
    assert os.path.isdir("./omniglot_resized")

    return


if __name__ == "__main__":
    main()