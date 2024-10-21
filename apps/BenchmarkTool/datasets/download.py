import download_tar_gz
import download_hdf5

def main():
  download_tar_gz.main() # fvecs datasets 
  download_hdf5.main() # hdf5 datasets

if __name__ == "__main__":
    main()