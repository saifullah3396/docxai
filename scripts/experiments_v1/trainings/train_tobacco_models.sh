#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# train models on tobacco pretrained on rvlcdip

# train nfnet on tobacco3482
./scripts/train.sh +train=tobacco3482 args/model_args=timm_model args.model_args.config.model_type=dm_nfnet_f1 args.training_args.experiment_name=dm_nfnet_f1_tobacco args.model_args.checkpoint_state_dict_key=model args.model_args.pretrained_checkpoint=/netscratch/saifullah/resources/pretrained_models/rvlcdip/dm_nfnet_f1.pt
./scripts/train.sh +train=tobacco3482 args/model_args=timm_model args.model_args.config.model_type=densenet121 args.training_args.experiment_name=densenet121_tobacco args.model_args.checkpoint_state_dict_key=model args.model_args.pretrained_checkpoint=/netscratch/saifullah/resources/pretrained_models/rvlcdip/densenet121.pt
./scripts/train.sh +train=tobacco3482 args/model_args=timm_model args.model_args.config.model_type=mobilenetv3_large_100 args.training_args.experiment_name=mobilenetv3_large_100_tobacco args.model_args.checkpoint_state_dict_key=model args.model_args.pretrained_checkpoint=/netscratch/saifullah/resources/pretrained_models/rvlcdip/mobilenetv3_large_100.pt
./scripts/train.sh +train=tobacco3482 args/model_args=timm_model args.model_args.config.model_type=res2net50_26w_8s args.training_args.experiment_name=res2net50_26w_8s_tobacco args.model_args.checkpoint_state_dict_key=model args.model_args.pretrained_checkpoint=/netscratch/saifullah/resources/pretrained_models/rvlcdip/res2net50_26w_8s.pt
./scripts/train.sh +train=tobacco3482 args/model_args=timm_model args.model_args.config.model_type=senet154 args.training_args.experiment_name=senet154_tobacco args.model_args.checkpoint_state_dict_key=model args.model_args.pretrained_checkpoint=/netscratch/saifullah/resources/pretrained_models/rvlcdip/senet154.pt