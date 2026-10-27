import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available! Found {num_gpus} GPU(s).")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
else:
    print("CUDA is not available. Using CPU.")