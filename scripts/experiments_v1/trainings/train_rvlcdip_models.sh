#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# train nfnet on rvlcdip
./scripts/train.sh +train=rvlcdip args/model_args=convnext_base args.training_args.experiment_name=convnext_rvlcdip
