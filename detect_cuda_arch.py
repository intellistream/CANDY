# detect_cuda_arch.py
import torch

def get_cuda_arch():
    if not torch.cuda.is_available():
        print("No CUDA device available")
        return

    capabilities = []
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        arch = f"{cap[0]}{cap[1]}"
        capabilities.append(arch)

    # Print the architectures as a semicolon-separated string (CMake will handle it)
    print(";".join(capabilities))

if __name__ == "__main__":
    get_cuda_arch()
