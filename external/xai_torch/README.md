# Guided Research - Influence of PPML Methods on XAI computation

**Type:**
Guided Research

**Topic:**
Influence of PPML Methods on XAI computation

## Abstract
The aim of this Guided Research is to explore the influence that PPML methods have, when applied during training of Deep Neural Networks, on the explainability of the same, through the application of post-hoc xAI methods like attribution or concept-based explanations.

### Research Questions
1. How does the application of PPML methods (e.g. different levels of differential privacy, federated learning, etc.) influence the usefulness of (post-hoc) xAI methods during inference?
	- Usefulness in the sense of:
		1. Fidelity - How well does the xai method reflect the actual decision making of the model?
		2. Sensitivity - How robust is the explanation against pertrubation of the input? / How dominant is the most relevant component, according to the xai method, on its own?
		3. Semantic usefulness - (Qualitative Measure) ==> How does the application of PPML affect the semantic coherence / intellegibility of explanations for the human user.

## Instructions
- Do **not** push data files (i.e. dataset, embeddings, models, large images etc.) to the repository as it is only intended for source files. When you want to share data files, use this [Cloud directory](https://cloud.dfki.de/owncloud/index.php/s/G8HBRfpqDnFK3Nb)
- As soon as a (sub)task is finished, push the code to the repository with self-explanatory details in the commit. Please do **not** push non-functional work in progress to the repository. Make sure that you use relative paths whenever possible, so that execution on another machine runs smoothly without requiring changes in the code.
- Regularly update the requirements file and use a fixed python environment dedicated to the project, so that the code remains executable after some time.
- If suitable, try to manage your code for a specific issue in a single jupyter notebook. If necessary you can write additional modules that you can import into the notebook. If you have to process the same notebook multiple times you can also export a .py file from the final notebook.
- Document the code along with your implementation because it becomes very difficult to do it at the end.
- Use the **"src"** folder for all of your code, **"plots"** folder for plots, **"notebooks"** for jupyter notebooks and **"results"** folder for any result reports. This base structure will help you to organize the files within the repository.
- Note down the main outline of your findings in this readme. This will help you to keep track of your contribution and already organise your final project presentation. Use the sections below that are created for seperate milestones of the project (we can define them flexibly). Only summarize the particular key findings. For details you can create individual files in the **"results"** folder and link them (e.g. results/M1_readme.md).

## Project plan

### Milestone 1 : [Training Baseline Models](#)

### Milestone 2 : [Applying XAI Methods on Public Models](#)

### Milestone 3 : [Training Models with Privacy](#)

### Milestone 4 : [Applying XAI Methods on Private Models](#)

### Milestone 5 : [Evaluation](#)

### Milestone 6: [Writing](#)

### Milestone 7: [Presentation](#)

## Related Literature
All reviewed related literature can be found [here](#).

## Environment Setup
Install the dependencies:
```
pip install -r requirements.txt
```

Setup environment variables:
```
export PYTHONPATH=<path/to/xai_torch>/src
export ROOT_DIR=/

# can be any directory where datasets are cached and model training outputs are generated.
export XAI_TORCH_OUTPUT_DIR=</your/output/dir>
```