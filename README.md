# The Reality of High Performing Deep Learning Models: A Case Study on Document Image Classification
This repository contains the code for the paper [The Reality of High Performing Deep Learning Models: A Case Study on
Document Image Classification] by Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please follow the steps below.

# Requirements
Please install the requirements with pip as follows:
```
pip install -r requirements.txt
```

Set PYTHONPATH to match source the directory:
```
export PYTHONPATH=`pwd`/src:`pwd`/external/xai_torch/src
```

Create output directory for holding dataset, models, etc
```
export ROOT_DIR=/<path/to/dataset>/
export XAI_TORCH_OUTPUT_DIR=</path/to/output/>
export XAI_TORCH_CACHE_DIR=</path/to/cache/>
mkdir -p $XAI_TORCH_OUTPUT_DIR
mkdir -p $XAI_TORCH_CACHE_DIR
```

# Run the analysis for generating AOPC metrics alexnet model on tobacco3482 dataset
```
./scripts/analyze.sh +analysis_v1=base_analysis/tobacco3482/alexnet
```

# Run the analysis for generating Sensitivity, Infidelity and Continuity metrics alexnet model on tobacco3482 dataset
```
./scripts/analyze.sh +analysis_v2=sens_inf_cont/tobacco3482/alexnet
```

# Citation
If you find this useful in your research, please consider citing our associated paper:
```
 @article{Saifullah_2023,
title={The Reality of High Performing Deep Learning Models: A Case Study on Document Image Classification},
url={http://dx.doi.org/10.36227/techrxiv.19310537.v3},
DOI={10.36227/techrxiv.19310537.v3},
publisher={Institute of Electrical and Electronics Engineers (IEEE)},
author={Saifullah, Saifullah and Agne, Stefan and Dengel, Andreas and Ahmed, Sheraz},
year={2023},
month=oct }
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.
