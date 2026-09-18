#!/bin/bash
#SBATCH --job-name=lstm_hourly_de
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=150000MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yh2352@partner.kit.edu
#SBATCH --output=logs/cudalstm_de1h_10ep_seed%a_%A.out
#SBATCH --error=logs/cudalstm_de1h_10ep_seed%a_%A.err
#SBATCH --array=100,200,300

# Each array task gets its own SLURM_ARRAY_TASK_ID (100, 200, or 300),
# since --array=100,200,300 lists the values explicitly rather than a range.
SEED=${SLURM_ARRAY_TASK_ID}
echo "Running seed: ${SEED}"

# Activate environment
module purge
module load jupyter/ai/2026-03-06
module unload devel/cuda/12.8

# Sanity checks
nvidia-smi
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('cuDNN version:', torch.backends.cudnn.version())"

python3 /pfs/data6/home/ka/ka_iwu/ka_yh2352/Hy2DL-Testing/examples/lstm_rainfall_runoff.py \
    --seed "${SEED}"