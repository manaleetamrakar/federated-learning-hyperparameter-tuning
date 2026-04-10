from pydantic import BaseModel
from typing import Optional

class SystemSpecs(BaseModel):
    ram_gb: float
    cpu_cores: int
    gpu_type: str  # e.g., "None", "Nvidia RTX 3060"
    vram_gb: float # 0 if None
    dataset_name: str # e.g., "mnist", "emnist"

class HyperparameterRecommendation(BaseModel):
    learning_rate: float
    epochs: int
    batch_size: int
    optimizer_name: str
    rationale: str

def recommend_hyperparameters(specs: SystemSpecs) -> HyperparameterRecommendation:
    # Rule-based logic
    
    # 1. Base batch size based on RAM and VRAM
    if specs.vram_gb >= 8:
        batch_size = 256
    elif specs.vram_gb >= 4:
        batch_size = 128
    elif specs.ram_gb >= 16:
        batch_size = 64
    else:
        batch_size = 32

    # 2. Adjust for dataset
    if specs.dataset_name.lower() == "emnist":
        # EMNIST is larger and more complex, needs more epochs, slightly lower batch size if constrained
        epochs = 8
        if batch_size > 64 and specs.vram_gb == 0:
            batch_size = 64 # Safety for CPU
    else:
        # MNIST is simple, trains fast
        epochs = 5
    
    # Fast GPU can afford more epochs if user wants higher accuracy, but we keep it reasonable
    if specs.vram_gb > 0:
        epochs = int(epochs * 1.5)
        
    # Optimizer and Learning Rate Rules
    optimizer_name = "adam"
    learning_rate = 0.001
    
    if batch_size >= 128:
        # Large batch sizes might benefit from slightly larger LR warmup, but 0.001 is safe for Adam.
        pass
    
    # Very weak system
    if specs.ram_gb <= 4 and specs.vram_gb == 0:
        optimizer_name = "rmsprop" # Sometimes lighter on memory depending on impl
        batch_size = 32
        epochs = 3
        
    # Generate rationale string
    rationale = f"Based on your system ({'GPU with ' + str(specs.vram_gb) + 'GB VRAM' if specs.vram_gb > 0 else 'CPU-only, ' + str(specs.ram_gb) + 'GB RAM'}), "
    rationale += f"a batch size of {batch_size} is optimal to avoid OOM errors. "
    rationale += f"We selected {epochs} epochs using {optimizer_name.capitalize()} for {specs.dataset_name}."

    return HyperparameterRecommendation(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        rationale=rationale
    )
