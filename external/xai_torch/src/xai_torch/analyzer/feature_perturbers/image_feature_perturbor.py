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
from ignite.engine import Engine, Events
from ignite.handlers.stores import EpochOutputStore
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
    attack: Callable,
    transforms: List[Callable],
    collate_fn: Callable,
    device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    visualize: bool = False
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

            # get inputs
            # batch size is always 1 for inputs
            perturbation_params, input, target = batch

            # apply attacks
            att_output = attack(
                images=input.repeat(len(perturbation_params), 1, 1, 1),
                perturbation_params=perturbation_params,
            )
            if isinstance(att_output, dict):
                att_images = att_output.pop("image")
            else:
                att_images = att_output

            if visualize:
                from matplotlib import pyplot as plt
                from torchvision.utils import make_grid

                if att_images.shape[1] == 1:
                    att_images = att_images.repeat(1, 3, 1, 1)
                att_images = att_images * 0.5 + 0.5
                image_grid = make_grid(att_images[:32].cpu(), nrow=4)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(image_grid.permute(1, 2, 0))
                plt.show()

            # generate attack batch
            att_batch = {DataKeys.IMAGE: att_images, DataKeys.LABEL: target.repeat(len(perturbation_params))}

            # now transform the images in the batch and apply collate fns to create tensors
            if collate_fn is not None:
                att_batch = [dict(zip(att_batch, t)) for t in zip(*att_batch.values())]
                if transforms is not None:
                    for sample in att_batch:
                        for transform in transforms:
                            sample = transform(sample)

                att_batch = collate_fn(att_batch)

            # put batch to device
            att_batch = convert_tensor(att_batch, device=device)

            # get output
            output = model.predict_step(att_batch)
            logits = output[DataKeys.LOGITS]

            # get softmax over outputs
            from torch.nn.functional import softmax

            # get conf scores
            conf_scores = softmax(logits)

            # get corrects
            corrects = logits.argmax(dim=1) == att_batch[DataKeys.LABEL]

            # forward pass
            return {"logits": logits, "conf_scores": conf_scores, "corrects": corrects, "att_output": att_output}

    return Engine(step)


class TerminateOnMissclassification:
    def __init__(self):
        self._output_transform = lambda x: x["corrects"]

    def __call__(self, engine: Engine) -> None:
        corrects = engine.state.output["corrects"]

        # get success index
        successful_index = -1
        list_false = (corrects == False).nonzero(as_tuple=True)[0]
        if len(list_false) > 0:
            successful_index = list_false[0].item()
        if successful_index != -1:
            engine.terminate_epoch()
            # print("Terminating engine run: ", engine.state.iteration)

        return engine.state.output


@dataclass
class FeaturePerturber:
    collate_fn: Callable
    transforms: Callable
    attr_map_grid_cell_size: int = 4
    attr_map_reduce_fn: str = "mean"
    max_perturbation_percentage: float = 1.0  # [0, ..., 1.0]
    perturbation_step_size: int = 1
    max_perturbation_steps: int = None
    eval_batch_size: int = 32
    force_gray_to_rgb: bool = False
    normalize_attributions: bool = True
    terminate_on_misclassification: bool = False
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
                    attr_map = viz._normalize_image_attr(attr_map.cpu().permute(1, 2, 0).numpy(), sign="positive")
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
        with torch.no_grad():
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
            attack_initializer = self.get_attack_initializer(
                total_regions=total_regions, importance_order=importance_order
            )

            # get perturbation list
            max_perturbation_steps = (
                self.max_perturbation_steps + 1 if total_regions > self.max_perturbation_steps else total_regions + 1
            )
            perturbation_params = torch.arange(0, max_perturbation_steps, self.perturbation_step_size)

            perturbation_results_list = []
            for attr_map_grid, input, target in zip(attr_map_grids, inputs, targets):
                # create the attack object
                perturbation_params_per_batch = torch.split(perturbation_params, self.eval_batch_size)
                attack = attack_initializer()
                attack.setup(attr_map_grid, input)

                feature_perturbation_engine = initialize_feature_perturbation_engine(
                    model,
                    attack=attack,
                    transforms=self.transforms,
                    collate_fn=self.collate_fn,
                    device=idist.device(),
                    visualize=False
                )

                epoch_metric = OutputGatherer(output_transform=lambda output: output[DataKeys.LOGITS])
                epoch_metric.attach(feature_perturbation_engine, "logits")

                epoch_metric = OutputGatherer(output_transform=lambda output: output["conf_scores"])
                epoch_metric.attach(feature_perturbation_engine, "conf_scores")

                epoch_metric = OutputGatherer(output_transform=lambda output: output["corrects"])
                epoch_metric.attach(feature_perturbation_engine, "corrects")

                epoch_metric = EpochOutputStore(output_transform=lambda output: output["att_output"])
                epoch_metric.attach(feature_perturbation_engine, "att_output")

                if self.terminate_on_misclassification:
                    feature_perturbation_engine.add_event_handler(
                        Events.ITERATION_COMPLETED, TerminateOnMissclassification()
                    )

                def dataloader():
                    for params in perturbation_params_per_batch:
                        yield params, input, target

                output = feature_perturbation_engine.run(dataloader())

                # get outputs
                logits, conf_scores, corrects, att_outputs = (
                    output.metrics["logits"],
                    output.metrics["conf_scores"],
                    output.metrics["corrects"],
                    feature_perturbation_engine.state.att_output,
                )

                # unroll att_outputs
                if isinstance(att_outputs, dict):
                    att_outputs = {k: torch.cat([dic[k] for dic in att_outputs]) for k in att_outputs[0]}
                elif isinstance(att_outputs, list):
                    att_outputs = {"att_outputs": torch.cat(att_outputs)}
                else:
                    att_outputs = {"att_outputs": att_outputs[0]}

                # get conf score of target class
                conf_score = conf_scores[:, target].cpu().numpy()

                # get success index
                successful_index = -1
                list_false = (corrects == False).nonzero(as_tuple=True)[0]
                if len(list_false) > 0:
                    successful_index = list_false[0].item()

                if successful_index != -1:
                    success_conf_score = conf_score[successful_index]
                    success_param = perturbation_params[successful_index].item()
                    percent_perturbation = success_param / total_regions

                    pr = PerturbationResults(
                        success_conf_score=success_conf_score,
                        success_param=success_param,
                        percent_perturbation=percent_perturbation,
                        conf_score=conf_score.tostring(),
                        missclassified=True,
                    )
                else:
                    pr = PerturbationResults(
                        success_conf_score=-1,
                        success_param=-1,
                        percent_perturbation=-1,
                        conf_score=conf_score.tostring(),
                        missclassified=False,
                    )

                # convert pr to dict
                pr = dataclasses.asdict(pr)

                # attach additional outputs
                att_outputs = {k: v[successful_index] for k, v in att_outputs.items()}
                for k, v in att_outputs.items():
                    pr[k] = v.cpu().numpy()

                perturbation_results_list.append(pr)
            return perturbation_results_list
