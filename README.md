# DFL Face Extractor for ComfyUI

Automated face extraction from video/images with reference-based matching. Designed for DeepFaceLab de-aging and face swap workflows.

## Features

- **Reference-based face matching**: Provide reference image(s) of your target character, automatically extract matching faces from footage
- **Video and image sequence support**: Process MP4/AVI/MOV files or image sequences
- **DFL-compatible output**: Generates aligned faces and masks in DeepFaceLab format
- **InsightFace backend**: High-accuracy face detection and recognition using ArcFace embeddings
- **Configurable thresholds**: Fine-tune similarity matching for your specific use case

## Installation

### 1. Clone to ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/comfyui_dfl_extractor.git
# Or copy the folder manually
```

### 2. Install dependencies

```bash
cd comfyui_dfl_extractor
pip install -r requirements.txt

# For GPU acceleration (recommended)
pip install onnxruntime-gpu

# Or CPU only
pip install onnxruntime
```

### 3. Download InsightFace models (automatic on first run)

The buffalo_l model will download automatically. For manual download:
```bash
# Models are cached in ~/.insightface/models/
```

## Nodes

### DFL Reference Face Embedding

Computes face embeddings from reference images.

**Inputs:**
- `reference_images`: One or more images of the target character
- `detection_backend`: "insightface" (recommended) or "opencv_cascade"
- `min_confidence`: Minimum face detection confidence (0.1-1.0)

**Outputs:**
- `reference_embedding`: Embedding data for matching

**Tips:**
- Use 5-10 reference images with varied expressions and lighting
- Include front-facing and slight angle shots
- Avoid heavily occluded faces

### DFL Video Face Extractor

Extracts matching faces from video files.

**Inputs:**
- `video_path`: Path to video file (MP4, AVI, MOV, etc.)
- `reference_embedding`: From DFL Reference Face Embedding node
- `output_directory`: Where to save extracted faces
- `similarity_threshold`: How closely face must match (0.3-0.95, default 0.6)
- `frame_skip`: Process every Nth frame (1 = all frames)
- `margin_factor`: Extra padding around face (0.1-1.0, default 0.4)
- `output_size`: Output image size in pixels (default 512)
- `max_faces_per_frame`: Maximum matching faces to extract per frame

**Outputs:**
- `output_path`: Directory containing extracted faces
- `extracted_count`: Number of faces extracted
- `preview_grid`: Preview of extracted faces

### DFL Image Sequence Extractor

Same as video extractor but for image sequences (batched IMAGE input).

### DFL Face Matcher

Compare two faces and get similarity score. Useful for tuning thresholds.

### DFL Batch Saver

Save extracted faces in DFL-compatible directory structure.

## Example Workflows

### Basic Video Extraction

```
[Load Image] → [DFL Reference Face Embedding] → [DFL Video Face Extractor]
                                                          ↓
                                                 [Preview Image]
```

### Multi-Reference Extraction (More Robust)

```
[Load Image Batch (5-10 refs)] → [DFL Reference Face Embedding] → [DFL Video Face Extractor]
                                           ↓
                                  (averaged embedding for
                                   better matching)
```

### Threshold Tuning Workflow

```
[Load Reference] → [DFL Reference Face Embedding] ─┐
                                                   ├→ [DFL Face Matcher] → [Display Similarity]
[Load Test Image] ─────────────────────────────────┘
```

## Directory Structure Output

```
output_directory/
├── aligned/
│   ├── 00000000_0.png
│   ├── 00000001_0.png
│   └── ...
├── masks/
│   ├── 00000000_0_mask.png
│   └── ...
├── debug/
│   └── extraction_log.json
```

## Tips for Best Results

1. **Reference Images**: Use 5-10 clear images of the target face
2. **Similarity Threshold**: 
   - Start at 0.6 for general use
   - Lower (0.4-0.5) for varied expressions/lighting
   - Higher (0.7-0.8) for strict matching
3. **Frame Skip**: Use 2-5 for quick preview, 1 for final extraction
4. **Margin Factor**: 0.4 works well for DFL, increase to 0.6 for more context

---

# Step 2: Distributed Training for DeepFaceLab

## Overview

Your idea of distributed training with weight averaging is valid but requires careful implementation. Here are the approaches:

## Approach 1: Data Parallel Training (Recommended)

Run the same model across multiple GPUs processing different batches simultaneously.

### Implementation for DFL:

DFL's training scripts would need modification. Here's the concept:

```python
# distributed_dfl_trainer.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_distributed(rank, world_size, model, dataset):
    setup_distributed(rank, world_size)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Training loop - gradients automatically synchronized
    for batch in dataloader:
        loss = model(batch)
        loss.backward()  # Gradients averaged across GPUs
        optimizer.step()
```

**Pros:**
- Near-linear speedup
- Gradients synchronized every step
- Native PyTorch support

**Cons:**
- Requires all GPUs on same network
- DFL architecture needs modification

## Approach 2: Federated/Periodic Averaging (Your Suggestion)

Train separate models, periodically average weights.

```python
# federated_averaging.py
import torch
import numpy as np
from pathlib import Path

def average_model_weights(model_paths: list, output_path: str):
    """
    Average weights from multiple trained models.
    Works with DFL .npy weight files or PyTorch state dicts.
    """
    state_dicts = []
    
    for path in model_paths:
        if path.endswith('.npy'):
            # DFL format
            weights = np.load(path, allow_pickle=True).item()
            state_dicts.append(weights)
        else:
            # PyTorch format
            state_dicts.append(torch.load(path, map_location='cpu'))
    
    # Average all weights
    averaged = {}
    keys = state_dicts[0].keys()
    
    for key in keys:
        weights = [sd[key] for sd in state_dicts]
        if isinstance(weights[0], np.ndarray):
            averaged[key] = np.mean(weights, axis=0)
        elif isinstance(weights[0], torch.Tensor):
            averaged[key] = torch.stack(weights).mean(dim=0)
        else:
            # Non-tensor params (take first)
            averaged[key] = weights[0]
    
    # Save averaged model
    if output_path.endswith('.npy'):
        np.save(output_path, averaged)
    else:
        torch.save(averaged, output_path)
    
    return averaged


def federated_training_coordinator(
    machines: list,  # List of machine configs
    sync_interval: int = 1000,  # Steps between syncs
    target_loss: float = 0.02,
    max_iterations: int = 100
):
    """
    Coordinate training across multiple machines.
    """
    iteration = 0
    
    while iteration < max_iterations:
        # 1. Each machine trains for sync_interval steps
        for machine in machines:
            machine.train(steps=sync_interval)
        
        # 2. Collect weights and losses from all machines
        model_paths = [m.get_current_weights_path() for m in machines]
        losses = [m.get_current_loss() for m in machines]
        
        print(f"Iteration {iteration}, Losses: {losses}")
        
        # 3. Check convergence
        avg_loss = np.mean(losses)
        if avg_loss < target_loss:
            print(f"Converged at iteration {iteration}")
            break
        
        # 4. Average weights
        averaged_path = f"averaged_model_iter_{iteration}.npy"
        average_model_weights(model_paths, averaged_path)
        
        # 5. Distribute averaged weights back to all machines
        for machine in machines:
            machine.load_weights(averaged_path)
        
        iteration += 1


# Example usage with DFL weight files
if __name__ == "__main__":
    # Paths to models trained on different machines
    model_paths = [
        "/machine1/workspace/model/encoder.npy",
        "/machine2/workspace/model/encoder.npy", 
        "/machine3/workspace/model/encoder.npy",
    ]
    
    # Average them
    average_model_weights(model_paths, "averaged_encoder.npy")
```

### Practical Implementation for DFL:

1. **Setup**: Same dataset on all machines
2. **Training**: Each machine runs `train.bat` independently
3. **Sync Script**: Periodically average weights:

```python
# sync_dfl_models.py
import subprocess
import time
from pathlib import Path
import numpy as np

MACHINES = [
    {"host": "machine1", "workspace": "/dfl/workspace"},
    {"host": "machine2", "workspace": "/dfl/workspace"},
]

SYNC_INTERVAL_MINUTES = 60
MODEL_FILES = ["encoder.npy", "inter_AB.npy", "decoder_src.npy", "decoder_dst.npy"]

def sync_and_average():
    while True:
        print(f"Syncing models...")
        
        for model_file in MODEL_FILES:
            paths = []
            
            # Download from all machines
            for i, machine in enumerate(MACHINES):
                local_path = f"/tmp/model_{i}_{model_file}"
                remote_path = f"{machine['workspace']}/model/{model_file}"
                
                subprocess.run([
                    "scp", f"{machine['host']}:{remote_path}", local_path
                ])
                paths.append(local_path)
            
            # Average
            weights = [np.load(p, allow_pickle=True).item() for p in paths]
            averaged = {}
            for key in weights[0].keys():
                arrays = [w[key] for w in weights]
                if isinstance(arrays[0], np.ndarray):
                    averaged[key] = np.mean(arrays, axis=0)
                else:
                    averaged[key] = arrays[0]
            
            # Save and distribute back
            avg_path = f"/tmp/averaged_{model_file}"
            np.save(avg_path, averaged)
            
            for machine in MACHINES:
                remote_path = f"{machine['workspace']}/model/{model_file}"
                subprocess.run([
                    "scp", avg_path, f"{machine['host']}:{remote_path}"
                ])
        
        print(f"Sync complete. Waiting {SYNC_INTERVAL_MINUTES} minutes...")
        time.sleep(SYNC_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    sync_and_average()
```

## Approach 3: Gradient Checkpointing + Accumulation

If single-machine with one GPU, simulate larger batches:

```python
# Accumulate gradients over multiple forward passes
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Important Caveats for Weight Averaging

1. **Same Initialization**: All models MUST start from identical weights
2. **Same Hyperparameters**: Learning rate, batch size, etc. should match
3. **Loss Landscape**: Averaging works best when models stay in similar loss basins
4. **Sync Frequency**: Too frequent = overhead, too rare = divergent models

### Recommended Sync Schedule for DFL:

| Training Stage | Sync Interval |
|---------------|---------------|
| Early (high loss) | Every 5000 iterations |
| Mid (loss dropping) | Every 2000 iterations |
| Late (fine-tuning) | Every 1000 iterations |

## Monitoring Convergence

```python
# Monitor that models stay synchronized
def check_model_divergence(model_paths):
    embeddings = []
    for path in model_paths:
        weights = np.load(path, allow_pickle=True).item()
        # Flatten all weights into single vector
        flat = np.concatenate([w.flatten() for w in weights.values()])
        embeddings.append(flat)
    
    # Compute pairwise distances
    from itertools import combinations
    for i, j in combinations(range(len(embeddings)), 2):
        dist = np.linalg.norm(embeddings[i] - embeddings[j])
        print(f"Model {i} vs {j} distance: {dist}")
        if dist > THRESHOLD:
            print("WARNING: Models diverging significantly!")
```

---

## License

MIT License
