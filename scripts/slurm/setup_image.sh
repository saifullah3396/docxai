srun --gpus-per-task=0 \
  --container-image=/netscratch/$USER/envs/xai_torch.sqsh \
  --container-workdir=/home/$USER/projects/colorized/docxclassifier \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/:/home/$USER/ \
  --container-save=/netscratch/$USER/envs/xai_torch_v2.sqsh \
  --partition V100-16GB \
  --pty /bin/bash
