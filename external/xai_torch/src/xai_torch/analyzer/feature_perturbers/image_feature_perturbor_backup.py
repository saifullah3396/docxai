"""
Defines the feature attribution generation task.
"""

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import ignite.distributed as idist
import numpy as np
import torch
from captum.attr import visualization as viz
from xai_torch.analyzer.feature_perturbers.attacks.factory import AttacksFactory
from xai_torch.core.constants import DataKeys
from xai_torch.core.training.metrics.output_gatherer import OutputGatherer


@dataclasses.dataclass
class PerturbationResults:
    success_conf_score: float
    success_param: float
    conf_score: List[float]
    percent_perturbation: float
    missclassified: bool


def initialize_feature_perturbation_engine(
    model: torch.nn.Module,
    attack_initializer: Callable,
    transforms: List[Callable],
    collate_fn: Callable,
    device: Optional[Union[str, torch.device]] = torch.device("cpu"),
) -> Callable:
    from ignite.engine import Engine

    def step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        """
        Define the computation step
        """

        with torch.no_grad():
            from ignite.utils import convert_tensor

            # ready model for evaluation
            model.eval()

            # now transform the images in the batch and apply collate fns to create tensors
            samples_list = [dict(zip(batch, t)) for t in zip(*batch.values())]
            for sample in samples_list:
                for transform in transforms:
                    sample = transform(sample)
            batch = collate_fn(samples_list)

            # put batch to device
            batch = convert_tensor(batch, device=device)

            # get output
            output = model.predict_step(batch)
            logits = output[DataKeys.LOGITS]

            # get softmax over outputs
            from torch.nn.functional import softmax

            # get conf scores
            conf_scores = softmax(logits)

            # get corrects
            corrects = logits.argmax(dim=1) == batch[DataKeys.LABEL]

            # forward pass
            return {"logits": logits, "conf_scores": conf_scores, "corrects": corrects}

    return Engine(step)


@dataclass
class FeaturePerturber:
    collate_fn: Callable
    transforms: Callable
    attr_map_grid_cell_size: int = 4
    attr_map_reduce_fn: str = "mean"
    max_perturbation_percentage: float = 1.0  # [0, ..., 1.0]
    perturbation_step_size: int = 1
    max_perturbation_steps: int = 100
    eval_batch_size: int = 32
    force_gray_to_rgb: bool = False
    normalize_attributions: bool = True
    attack_type: str = "basic_attack"
    attack_config: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        assert self.attr_map_reduce_fn in ["sum", "mean"]

    @abstractmethod
    def run(
        self,
        model: torch.nn.Module,
        inputs,
        targets,
        attr_maps,
        importance_order="descending",
    ):
        pass


class PerturbationDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, attr_map_grids, perturbation_params, attack_initializer) -> None:
        super().__init__()
        self._inputs = inputs
        self._targets = targets
        self._attr_map_grids = attr_map_grids
        self._perturbation_params = perturbation_params
        self._attack_initializer = attack_initializer

    @property
    def base_batch_size(self):
        return self._base_batch_size

    @property
    def perturbation_len(self):
        return self._perturbation_len

    def setup(self):
        # generate attack
        attack = self._attack_initializer()
        attack.setup(self._attr_map_grids)

        self._perturbation_len = len(self._perturbation_params)
        self._base_batch_size = self._inputs.shape[0]
        self._perturbation_params = self._perturbation_params.repeat(self._base_batch_size)

        self._inputs = torch.repeat_interleave(self._inputs, self._perturbation_len, dim=0)
        self._inputs = attack(
            inputs=self._inputs,
            perturbation_params=self._perturbation_params,
            perturbation_len=self._perturbation_len,
        )
        self._targets = torch.repeat_interleave(self._targets, self._perturbation_len, dim=0)

        # visualize for debugging
        # self.visualize()

    def visualize(self):
        # visualize perturbations here
        import matplotlib.pyplot as plt

        for idx, att_image in enumerate(self._inputs.reshape(self._base_batch_size, -1, *self._inputs.shape[1:])):
            from matplotlib import pyplot as plt
            from torchvision.transforms.functional import resize
            from torchvision.utils import make_grid

            if att_image.shape[1] == 1:
                att_image = att_image.repeat(1, 3, 1, 1)
            att_image = torch.cat(
                [
                    att_image,
                    resize(self._attr_map_grids[idx].unsqueeze(0), [*att_image.shape[2:]])
                    .unsqueeze(0)
                    .repeat(1, 3, 1, 1),
                ]
            )
            image_grid = make_grid(att_image[:64], nrow=4)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(image_grid.permute(1, 2, 0))
            plt.show()

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):
        return {DataKeys.IMAGE: self._inputs[idx], DataKeys.LABEL: self._targets[idx]}


@dataclass
class ImageFeaturePerturber(FeaturePerturber):
    def get_attr_map_grids(self, attr_maps):
        kx = self.attr_map_grid_cell_size
        ky = self.attr_map_grid_cell_size

        attr_map_grids = []
        for attr_map in attr_maps:
            # normalize if required
            if self.normalize_attributions:
                # first we normalize the attribution maps for positive contributions
                try:
                    attr_map = viz._normalize_image_attr(attr_map.cpu().permute(1, 2, 0).numpy(), sign="all")
                except Exception as e:
                    c, h, w = list(attr_map.cpu().shape)
                    attr_map = np.zeros((h, w))
                attr_map = torch.from_numpy(attr_map)
            else:
                # reduce map over channels
                attr_map = attr_map.sum(dim=0)

            # now we reduce the normalized map into a grid 224x224 -> 4x4 cell -> 56x56
            if self.attr_map_reduce_fn == "sum":
                # take sum per grid cell
                attr_map_grids.append(attr_map.unfold(0, ky, kx).unfold(1, ky, kx).sum(dim=(2, 3)))
            elif self.attr_map_reduce_fn == "mean":
                # take mean per grid cell
                attr_map_grids.append(attr_map.unfold(0, ky, kx).unfold(1, ky, kx).mean(dim=(2, 3)))

        return torch.stack(attr_map_grids)

    def get_attack_initializer(self, total_regions: int, importance_order: str) -> Callable:
        # total_perturbation_steps is the max number of steps we want to perform perturbation for.
        # this max step can either be set by defining arg_max directly or by giving how much percentage of
        # bins are to be removed.
        if self.max_perturbation_steps is None:
            self.max_perturbation_steps = int(total_regions * self.max_perturbation_percentage)

        # get attack configuration
        attack_type = self.attack_type
        attack_config = {
            "grid_cell_size": self.attr_map_grid_cell_size,
            "return_perturbation_output": True,
            "importance_order": importance_order,
            **self.attack_config,
        }

        # add some info
        # logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        # logger.info(f"Total regions: {total_regions}")
        # logger.info(f"Total perturbation steps to perform: {self.max_perturbation_steps}")
        # logger.info(f"Running attack type: {attack_type}")
        # logger.info(f"Attack configuration:\n {json.dumps(attack_config, indent=2)}")

        def create_attack():
            return AttacksFactory.create(attack_type, **attack_config)

        return create_attack

    def run(
        self,
        model: torch.nn.Module,
        inputs,
        targets,
        attr_maps,
        importance_order="descending",
    ):
        from torch.utils.data import DataLoader
        from torchvision import transforms

        # first we generate a grid over the original attribution map. for example for a grid with cell size 4,
        # a 224x224 map is converted to a 56x56 grid by taking mean/sum defined by attr_map_reduce_fn
        # over each cell. Defaults to mean.
        attr_map_grids = self.get_attr_map_grids(attr_maps)

        # resize images to the size of model input for perturbation if not
        if list(attr_maps.shape[2:]) != list(inputs.shape[2:]):
            resize_tf = torch.nn.Sequential(
                transforms.Resize(list(attr_maps.shape[2:])),
            )
            inputs = resize_tf(inputs)

        # total perturbation_regions. This is the total bins on the image that can be perturbed
        total_regions = attr_map_grids.shape[1] * attr_map_grids.shape[2]

        # initialize attack
        attack_initializer = self.get_attack_initializer(total_regions=total_regions, importance_order=importance_order)

        # get perturbation list
        max_perturbation_steps = (
            self.max_perturbation_steps + 1 if total_regions > self.max_perturbation_steps else total_regions + 1
        )
        perturbation_params = torch.arange(0, max_perturbation_steps, self.perturbation_step_size)
        dataset = PerturbationDataset(inputs, targets, attr_map_grids, perturbation_params, attack_initializer)
        dataset.setup()

        dataloader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
        )

        feature_perturbation_engine = initialize_feature_perturbation_engine(
            model,
            attack_initializer=attack_initializer,
            transforms=self.transforms,
            collate_fn=self.collate_fn,
            device=idist.device(),
        )

        epoch_metric = OutputGatherer(output_transform=lambda output: output[DataKeys.LOGITS])
        epoch_metric.attach(feature_perturbation_engine, "logits")

        epoch_metric = OutputGatherer(output_transform=lambda output: output["conf_scores"])
        epoch_metric.attach(feature_perturbation_engine, "conf_scores")

        epoch_metric = OutputGatherer(output_transform=lambda output: output["corrects"])
        epoch_metric.attach(feature_perturbation_engine, "corrects")

        output = feature_perturbation_engine.run(dataloader)

        # define output reshape func
        reshape_output = lambda x: x.reshape(dataset.base_batch_size, dataset.perturbation_len, -1)

        # get metrics
        logits, conf_scores, corrects = (
            output.metrics["logits"],
            output.metrics["conf_scores"],
            output.metrics["corrects"],
        )
        print(corrects)

        # reshape outputs for each base image
        logits = reshape_output(logits)
        conf_scores = reshape_output(conf_scores)
        corrects = reshape_output(corrects)

        # get success index
        successful_indices = []
        for correct in corrects:
            list_false = (correct == False).nonzero(as_tuple=True)[0]
            if len(list_false) > 0:
                successful_indices.append(list_false[0].item())
            else:
                successful_indices.append(-1)

        # get success confdence scores for target label
        perturbation_results_list = []
        print("targets", targets)
        for idx, conf_score in enumerate(conf_scores):
            conf_score = conf_score[:, targets[idx]].cpu().numpy()
            print("conf_score", conf_score)
            if successful_indices[idx] != -1:
                success_conf_score = conf_score[successful_indices[idx]]
                success_param = perturbation_params[successful_indices[idx]].item()
                percent_perturbation = success_param / total_regions

                pr = PerturbationResults(
                    success_conf_score=success_conf_score,
                    success_param=success_param,
                    percent_perturbation=percent_perturbation,
                    conf_score=conf_score,
                    missclassified=True,
                )
            else:
                pr = PerturbationResults(
                    success_conf_score=-1,
                    success_param=-1,
                    percent_perturbation=-1,
                    conf_score=conf_score,
                    missclassified=False,
                )
            perturbation_results_list.append(pr)
        print(perturbation_results_list)
        exit()
        return perturbation_results_list

    # def visualize_perturbed_images(self, perturbed_images, perturbation_outputs, show=False):
    #     rows = int(len(perturbed_images) / 4)
    #     cols = 4
    #     fig, gs = get_matplotlib_grid(rows, cols, figsize=8)

    #     # create rows for grid
    #     for idx, image in enumerate(perturbed_images):
    #         ax = plt.subplot(gs[0, idx])
    #         image = cv2.cvtColor(image.cpu().permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    #         ax.imshow(image)
    #         ax.set_xticks([])
    #         ax.set_yticks([])

    #         # add title
    #         title = ""
    #         title += f"\% Drop = {self.vis_perturbation_outputs[idx].percent_perturbation:.2f}\n"
    #         title += f"Missclassified = {self.vis_perturbation_outputs[idx].missclassified}\n"
    #         title += f"Initial Score = {self.vis_perturbation_outputs[idx].all_confidence_scores[0]:.2f}\n"
    #         if self.vis_perturbation_outputs[idx].success_confidence_score is not None:
    #             title += f"Final Score = {self.vis_perturbation_outputs[idx].success_confidence_score:.2f}\n"
    #         else:
    #             title += f"Final Score = {self.vis_perturbation_outputs[idx].all_confidence_scores[-1]:.2f}\n"
    #         ax.set_title(title)
    #         ax.imshow(image)

    #     if show:
    #         plt.show()

    #     return fig
