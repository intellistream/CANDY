import download_hdf5
import download_tar_gz

print(download_hdf5.__file__)


def main():
    download_tar_gz.main()  # fvecs datasets
    download_hdf5.main()  # hdf5 datasets


if __name__ == "__main__":
    main()
