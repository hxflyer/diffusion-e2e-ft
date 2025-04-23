import torch

def main():
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is available: {cuda_available}")
    
    # If CUDA is available, print additional information
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    main()
