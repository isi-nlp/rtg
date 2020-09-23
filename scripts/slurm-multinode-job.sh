#!/usr/bin/env bash 

#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1 --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:2
#SBATCH --output=R-%x.out.%j
#SBATCH --error=R-%x.err.%j
#SBATCH --export=NONE

# Pipeline script for MT
#
# Author = Thamme Gowda (tg@isi.edu)
# Date = April 3, 2019


RTG_PATH=~/sc2/repos/rtg

# 6 cpus max due to sqlite writes as bottleneck
export RTG_CPUS=4 #$SLURM_CPUS_ON_NODE # 120
export OMP_NUM_THREADS=$RTG_CPUS
export MKL_NUM_THREADS=$RTG_CPUS
export FP16="--fp16"
#defaults
CONDA_ENV=rtg     # empty means don't activate environment
OUT=$1

source ~/.bashrc

exit_log(){ echo "$2";  exit $1; }
usage() { echo "Usage: $0 -d <exp/dir>" 1>&2 ; exit 1; }


while getopts ":d:" o; do
    case "${o}" in
        d) OUT=${OPTARG} ;;
        *) usage ;;
    esac
done

[[ -n $OUT ]] || usage   # show usage and exit

# rank is the index of current host in list
NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)

echo "Output dir = $OUT"
[[ -d $OUT ]] || exit_log 2 "ERROR: $OUT not found"


conda activate $CONDA_ENV
[[ -f $OUT/rtg.zip ]] || exit_log 2 "$OUT/rtg.zip not found"
export PYTHONPATH=$OUT/rtg.zip


# assume all nodes have same number of GPUs
gpus_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
nodes=( $(scontrol show hostnames) )
tot_nodes="${#nodes[@]}"
master="${nodes[0]}"

node=$(hostname -s)

for rank in "${!nodes[@]}"; do
  if [[ "$node" == "${nodes[$rank]}" ]]; then
      cmd="python -m rtg.distrib.launch -N $tot_nodes -r $rank -P $gpus_per_node -G 1"
      cmd="$cmd --master-addr $master --master-port 29600"
      cmd="$cmd -m rtg.pipeline $OUT $FP16"
      echo "RUN:: $(hostname):: $cmd"
      eval $cmd || exit_log 10 "Job failed. Check logs"
   fi
done

if [[ -z "$cmd" ]]; then
    exit_log 4 "node name resolution failed. Job didnt run"
fi
    
echo "done"
