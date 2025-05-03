#!/bin/bash

# Export the current date and time for job labeling
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="RUControversial"
export JOB_NAME=train_"$LABEL"_"$DATE"

# Environment variables to optimize performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Create directories for logs and scratch data
mkdir -p /scratch/${USER}/logs/RUControversial
mkdir -p /scratch/${USER}/data/RUControversial

# Submit the job
sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch/${USER}/logs/RUControversial/$JOB_NAME.%j.o
#SBATCH -e /scratch/${USER}/logs/RUControversial/$JOB_NAME.%j.e
#SBATCH --requeue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

module purge
module use /projects/community/modulefiles
module load cuda/11.7

set -x

cd $HOME/RUControversial

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs439

python bert_model.py
EOT

# Submit job:
# sbatch jobscript.sh

# See logs:
cd /scratch/$USER/logs/RUControversial