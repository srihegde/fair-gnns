#!/bin/bash

#SBATCH --job-name=fair_baseline
#SBATCH --output=outfiles/dai_gcn.out.%j
#SBATCH --error=outfiles/dai_gcn.out.%j
#SBATCH --time=72:00:00
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --gres=gpu:1
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3



DATA_LIST=("pokec_n" "pokec_z" "nba")

EXPERIMENT="dai_baseline"
MODEL="GCN"
SENS_NUM=200

srun bash -c "hostname;"

source /fs/class-projects/fall2021/cmsc742/c742g002/FairGNN/venv/bin/activate

for i in ${DATA_LIST[@]}; do
        srun bash -c "echo """
        srun bash -c "echo Running dataset: $i"
        # srun bash -c "bash /fs/class-projects/fall2021/cmsc742/c742g002/FairGNN/src/scripts/$i/train_fair${MODEL}.sh"
        srun bash -c "python train_varfair.py --seed=42 --epochs=2000 --model=$MODEL --sens_number=$SENS_NUM --dataset=$i --num-hidden=128 --acc=0.69 --roc=0.76 --alpha=100 --beta=1"
        # srun bash -c "python train_fairGNN.py --seed=42 --epochs=2000 --model=$MODEL --sens_number=$SENS_NUM --dataset=$i --num-hidden=128 --acc=0.69 --roc=0.76 --alpha=100 --beta=1"
done

deactivate
