#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

IMAGE=/netscratch/$USER/envs/xai_torch_v5.sqsh
WORK_DIR=$SCRIPT_DIR/../../
MOUNTS=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/ds-sds:/ds-sds:ro,/home/$USER/:/home/$USER/
CACHE_DIR=/netscratch/$USER/cache
PYTHON_PATH=$WORK_DIR/src:$WORK_DIR/external/xai_torch/src:$WORK_DIR/external/xai_torch/external/shap:$WORK_DIR/external/xai_torch/external/ocrodeg
EXPORTS="TERM=linux,NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,USER_DIR=$USER,XAI_TORCH_CACHE_DIR=$CACHE_DIR,XAI_TORCH_OUTPUT_DIR=/netscratch/$USER/xai_torch,PYTHONPATH=$PYTHON_PATH,TORCH_HOME=$CACHE_DIR/pretrained"
NODES=1
TASKS=1
GPUS_PER_TASK=1
CPUS_PER_TASK=8
PARTITION=batch
MEMORY=60

usage()
{
    echo "Usage:"
    echo "./gpu_run.sh <cmd>"
    echo ""
    echo " --image: Container image to use. "
    echo " --work-dir: Path to work directory. "
    echo " --mounts: Directories to mount. "
    echo " --nodes : Number of nodes."
    echo " --tasks : Number of tasks per node."
    echo " --gpus_per_task : Number of GPUs per task."
    echo " --cpus_per_task : Number of GPUs per task."
    echo " -p | --partition : GPU partition"
    echo " -m | --memory : Total memory"
    echo " -h | --help : Displays the help"
    echo ""
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
        shift # past argument
        usage
        exit
        ;;
    --image)
        IMAGE="$2"
        shift # past argument
        shift # past value
        ;;
    --work-dir)
        WORK_DIR="$2"
        shift # past argument
        shift # past value
        ;;
    --mounts)
        MOUNTS="$MOUNTS,"$2""
        shift # past argument
        shift # past value
        ;;
    --nodes)
        NODES="$2"
        shift # past argument
        shift # past value
        ;;
    --tasks)
        TASKS="$2"
        shift # past argument
        shift # past value
        ;;
    --gpus-per-task )
        GPUS_PER_TASK="$2"
        shift # past argument
        shift # past value
        ;;
    --cpus-per-task )
        CPUS_PER_TASK=$2
        shift # past argument
        shift # past value
        ;;
    -p  | --partition )
        PARTITION=$2
        shift # past argument
        shift # past value
        ;;
    -m  | --memory )
        MEMORY=$2
        shift # past argument
        shift # past value
        ;;
    +experiment )
        POSITIONAL_ARGS+=("$1=$2") # save positional arg
        shift # past argument
        ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

MEMORY=$(($MEMORY * $TASKS))

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ $GPUS_PER_TASK == 0 ]; then
    srun \
        --container-image=$IMAGE \
        --container-workdir=$WORK_DIR \
        --container-mounts=$MOUNTS \
        --export=$EXPORTS \
        -K \
        --nodes=$NODES \
        --ntasks-per-node=$(($TASKS / $NODES)) \
        --ntasks=$TASKS \
        --cpus-per-task=$CPUS_PER_TASK \
        --mem="${MEMORY}G" \
        -p $PARTITION \
        --task-prolog="`pwd`/scripts/slurm/install.sh" \
        $@
else
    srun \
        --container-image=$IMAGE \
        --container-workdir=$WORK_DIR \
        --container-mounts=$MOUNTS \
        --export=$EXPORTS \
        -K \
        --nodes=$NODES \
        --ntasks-per-node=$(($TASKS / $NODES)) \
        --ntasks=$TASKS \
        --gpus-per-task=$GPUS_PER_TASK \
        --cpus-per-task=$CPUS_PER_TASK \
        --mem="${MEMORY}G" \
        -p $PARTITION \
        --gpu-bind=none \
        --task-prolog="`pwd`/scripts/slurm/install.sh" \
        $@
fi
