# 3DSPA Setup Guide for Killarney Compute Cluster

This guide covers running 3DSPA training, inference, and evaluation on the Killarney cluster.

See the [Alliance Getting Started documentation](https://docs.alliancecan.ca/) for details.

---

## Project Setup

### 1. Connect and Create Project Directory

```bash
ssh username@killarney.alliancecan.ca

# Create project directory
PROJNAME=3dspa
cd $(ls -d ~/projects/*/ | head -n 1)
mkdir -p "$USER/$PROJNAME"
cd "$USER/$PROJNAME"
```

### 2. Clone Repository

```bash
git clone https://github.com/bchandna/3dspa_code.git
cd 3dspa_code
```

### 3. Clone TAPVid-3D (required for evaluation) 
```bash
git clone https://github.com/google-deepmind/tapnet.git
export PYTHONPATH="${PYTHONPATH}:$(pwd)/tapnet"
```

Add this to your job scripts or add to `~/.bashrc` when working in the project.

### 4. Download 3dspa Checkpoint from `https://drive.google.com/file/d/1sd3_MuXDXw6TKbay2rh0EjHg3Pbe9RFr/view?usp=sharing`.

---

## Environment Setup


### 1. Load Modules and Create Virtual Environment

```bash
# Load Python and CUDA
module load python/3.12 cuda/12.2

# Create virtual environment
virtualenv --no-download "venv" --prompt "3dspa"

# Activate
source venv/bin/activate
```

Check all packages in `requirements.txt`:

```bash
while IFS= read -r line; do
  [[ -n "$line" && "$line" != \#* ]] && {
    package=$(echo "$line" | sed 's/[<>=!~].*//')
    echo "~~ $package ($line) ~~"
    avail_wheels "$package" 2>/dev/null || echo "Not available in wheelhouse"
    echo
  }
done < requirements.txt
```

### 3. Install Dependencies

**WandB fix:** Use `wandb==0.18.0` (not 0.19.6 from wheelhouse) due to a bug on Alliance Canada: [wandb#8966](https://github.com/wandb/wandb/issues/8966). Add or pin this in your requirements.

Create `requirements_killarney.txt` or adjust `requirements.txt` before installing:

```bash
# Pin wandb to avoid Alliance Canada bug
pip install wandb==0.18.0

# Install remaining packages (prioritizes local wheelhouse when available)
pip install -r requirements.txt --find-links https://pypi.org/simple/ --prefer-binary
```

For PyTorch with a specific CUDA version (if not using wheelhouse):

```bash
# Example: CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Run remaining requirements separately (--index-url blocks pypi.org)
pip install -r requirements.txt --find-links https://pypi.org/simple/ --prefer-binary
```

### 4. Install Additional Inference Dependencies

Install **CoTracker3** from `https://github.com/facebookresearch/co-tracker` and **VideoDepthAnything** from 

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
# Download checkpoint: depth_anything_vitb14.pth
# Place in checkpoints/ or specify via --vda_model_path
```

---

## Example SLURM Job Scripts

---

## Training

### Example: `train.slrm`

```bash
#!/bin/bash
#SBATCH --account=<YOUR_AIP_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --tasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --job-name=3dspa_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH -D /project/<project>/<user>/3dspa/3dspa_code

# Load modules and activate venv
module load python/3.12 cuda/12.2
source venv/bin/activate

# TAPVid-3D path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/tapnet"

# Create log dir
mkdir -p logs

# Run training
python train.py \
  --model_type=3dspa \
  --checkpoint_dir=./checkpoints/3dspa \
  --wandb_project=3dspa \
  --wandb_run_name=3dspa_killarney \
  --batch_size=64 \
  --learning_rate=1e-4 \
  --num_epochs=300 \
  --num_output_frames=150 \
  --use_dino=True \
  --use_depth=True
```

**Submit:**

```bash
AIP_ACCOUNT="$(ls -d ~/projects/*/ | head -n 1 | sed 's|/$||' | sed 's|.*/||')"
mkdir -p logs

sbatch \
  --account "$AIP_ACCOUNT" \
  --nodes 1 \
  --gres gpu:l40s:1 \
  --tasks-per-node 1 \
  --mem 128G \
  --cpus-per-task 16 \
  --time 2-00:00:00 \
  -D "$(pwd)" \
  train.slrm
```

---

## Inference

### Example: `inference.slrm`

```bash
#!/bin/bash
#SBATCH --account=<YOUR_AIP_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --tasks-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --job-name=3dspa_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH -D /project/<project>/<user>/3dspa/3dspa_code

module load python/3.12 cuda/12.2
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/tapnet"

python inference.py \
  --checkpoint_path=./checkpoints/3dspa/3dspa_ckpt.npz \
  --video_path=./data/example_video.mp4 \
  --output_dir=./inference_output \
  --use_dino=True \
  --use_depth=True \
  --num_query_points=512 \
  --num_support_tracks=2048 \
  --tracking_grid_size=64 \
  --vda_model_path=./checkpoints/depth_anything_vitb14.pth
```

---

## Evaluation (TAPVid-3D)

### Example: `evaluate.slrm`

```bash
#!/bin/bash
#SBATCH --account=<YOUR_AIP_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --tasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --job-name=3dspa_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH -D /project/<project>/<user>/3dspa/3dspa_code

module load python/3.12 cuda/12.2
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/tapnet"

# Ensure TAPVid-3D dataset is available at dataset_path
python evaluate_tapvid3d.py \
  --checkpoint_path=./checkpoints/3dspa/checkpoint_100000 \
  --dataset_path=./data/tapvid3d_dataset \
  --output_dir=./eval_results \
  --batch_size=8 \
  --use_dino=True \
  --use_depth=True \
  --depth_scalings=median,per_trajectory \
  --data_sources=drivetrack,adt,pstudio \
  --use_minival=True
```


---