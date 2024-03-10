"""
Defines the feature attribution generation task.
"""

import abc
from abc import abstractmethod
from pathlib import Path
from typing import List, Mapping, Union

import h5py
import numpy as np
import torch
from captum.metrics import infidelity, sensitivity_max
from ignite.engine import CallableEventWithFilter, Engine, EventEnum, Events, EventsList
from torch.utils.tensorboard import SummaryWriter
from xai_torch.analyzer.feature_attr_analysis.attribution_wrappers import (
    CaptumAttrWrapperBase,
)
from xai_torch.analyzer.feature_attr_analysis.constants import EVALUATORS_REGISTRY
from xai_torch.analyzer.feature_attr_analysis.utilities import update_dataset_at_indices
from xai_torch.analyzer.feature_attr_analysis.visualizer import ImageVisualizer
from xai_torch.analyzer.feature_perturbers.image_feature_perturbor import (
    FeaturePerturber,
)
from xai_torch.core.constants import DataKeys
from xai_torch.utilities.decorators import register_as_child


class EvaluatorKeys:
    ATTR_MAP = "attr_map"
    CONTINUITY = "continuity"
    INFIDELITY = "infidelity"
    SENSITIVITY = "sensitivity"
    FEATURE_PERTURBATION = "feature_perturbation"
    AOPC = "aopc"


class EvaluatorEvents(EventEnum):
    ATTR_MAP_COMPUTED = f"{EvaluatorKeys.ATTR_MAP}_computed"
    CONTINUITY_COMPUTED = f"{EvaluatorKeys.CONTINUITY}_computed"
    INFIDELITY_COMPUTED = f"{EvaluatorKeys.INFIDELITY}_computed"
    SENSITIVITY_COMPUTED = f"{EvaluatorKeys.SENSITIVITY}_computed"
    FEATURE_PERTURBATION_COMPUTED = f"{EvaluatorKeys.FEATURE_PERTURBATION}_computed"
    AOPC_COMPUTED = f"{EvaluatorKeys.AOPC}_computed"


def register_evaluator(reg_name: str = ""):
    return register_as_child(
        base_class_type=EvaluatorBase,
        registry=EVALUATORS_REGISTRY,
        reg_name=reg_name,
    )


class EvaluatorBase:
    def __init__(
        self, output_file: str, summary_writer: SummaryWriter, overwrite: bool = False, save_eval: bool = True
    ):
        self._output_file = output_file
        self._summary_writer = summary_writer
        self._overwrite = overwrite
        self._save_eval = save_eval

    def compute(
        self,
        engine: Engine,
    ):
        pass

    def fire_completion_event(self, engine: Engine):
        pass

    def finished(self, name, max_data_required: int):
        if self._overwrite:
            return False
        # read from h5 if this data is already computed
        if Path(self._output_file).exists():
            hf = h5py.File(self._output_file, "r")

            # check if required data is already computed
            if name in hf and len(hf[name]) == max_data_required:
                return True
            return False

    def write_data_to_hdf5(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.array
    ):
        if not self._save_eval:
            return

        data = getattr(engine.state, key)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        update_dataset_at_indices(
            hf, key, indices, data, (None, *data.shape[1:]), overwrite=self._overwrite
        )

    def read_data_from_hdf5(self, engine: Engine, name: str, output_file: str):
        # read from h5 if this data is already computed
        if Path(output_file).exists():
            hf = h5py.File(output_file, "r")

            # get data indices
            indices = np.array(engine.state.batch["index"])
            print("reading indices from", indices)

            # check if required data is already computed
            if name in hf and indices.max() < len(hf[name]):
                print('name', name)
                data = self.transform_data_on_load(hf[name][indices])

                print('attr_maps', data.shape, data)

                # close file
                hf.close()

                return data

    def transform_data_on_load(self, data):
        return data

    def __call__(self, engine: Engine, name: str) -> None:
        data = None
        if not self._overwrite:  # do not read data if overwrite is true
            data = self.read_data_from_hdf5(engine, name, self._output_file)
        if data is None:
            print("computing data for ", name)
            data = self.compute(engine)
        setattr(engine.state, name, data)
        self.fire_completion_event(engine)

    def attach(
        self,
        engine: Engine,
        name: str = "attr_map",
        event: Union[
            str, Events, CallableEventWithFilter, EventsList
        ] = Events.ITERATION_COMPLETED,
    ) -> None:
        if not hasattr(engine.state, name):
            setattr(engine.state, name, None)
        engine.add_event_handler(event, self, name)


@register_evaluator(EvaluatorKeys.ATTR_MAP)
class AttributionEvaluator(EvaluatorBase):
    def __init__(
        self,
        output_file: str,
        summary_writer: SummaryWriter,
        attr: CaptumAttrWrapperBase,
        baselines: torch.Tensor,
        visualize: bool = False,
        visualization_tag: str = "",
        overwrite: bool = False,
        output_transform=None,
        save_eval=True
    ):
        super().__init__(
            output_file=output_file, summary_writer=summary_writer, overwrite=overwrite, save_eval=save_eval
        )

        self._attr = attr
        self._baselines = baselines
        if output_transform is None:
            self._output_transform = lambda x: {
                key: val
                for key, val in x.items()
                if key in [DataKeys.IMAGE, DataKeys.PRED]
            }
        else:
            self._output_transform = output_transform
        self._visualize = visualize
        self._summary_writer = summary_writer
        self._visualized = False
        self._visualization_tag = visualization_tag
        self._max_vis = 10

    def transform_data_on_load(self, data):
        import ignite.distributed as idist

        return torch.from_numpy(data).to(idist.device())

    def visualize(self, images, attrs, preds):
        vis_attr = attrs[: self._max_vis].detach().cpu()
        vis_images = images[: self._max_vis].detach().cpu()
        vis_preds = preds[: self._max_vis].detach().cpu()
        if not self._visualized:
            import matplotlib.pyplot as plt

            fig = ImageVisualizer.visualize_image_attr_grid(
                vis_attr, vis_images, vis_preds
            )
            self._summary_writer.add_figure(
                f"{self._visualization_tag}/{str(self._attr.__class__.__name__)}", fig
            )
            plt.close(fig)
        self._visualized = True

    def compute(self, engine: Engine) -> None:
        inputs = self._output_transform(engine.state.output)

        if "image" in inputs:
            inputs["inputs"] = inputs["image"]
            del inputs["image"]
        if "pred" in inputs:
            inputs["target"] = inputs["pred"]
            del inputs["pred"]

        if "baseline" in inputs:
            attrs = self._attr.attribute(
                **inputs,
            )
        else:
            attrs = self._attr.attribute(
                **inputs,
                baselines=self._baselines,
            )
        if self._visualize:
            self.visualize(inputs["inputs"], attrs, inputs["target"])

        return attrs

    def fire_completion_event(self, engine: Engine):
        engine.fire_event(EvaluatorEvents.ATTR_MAP_COMPUTED)


def normalize(attr, std_mean=True, per_sample=True):  # TODO added std_mean
    try:
        n_samples = attr.size()[0]
        is_tuple = False
        attr_t = attr
    except:
        attr_t = torch.stack(list(attr))
        is_tuple = True
        n_samples = attr_t.size()[0]
    if per_sample:
        tmp = []
        for i in range(n_samples):
            if std_mean:
                norm_attr = (attr_t[i] - torch.mean(attr_t[i])) / torch.std(attr_t[i])
            else:
                norm_attr = (attr_t[i] - attr_t[i].min()) / (attr_t[i].max() - attr_t[i].min())
            tmp.append(norm_attr)
        tmp = torch.stack(tmp)
    else:
        if std_mean:
            tmp = (attr_t - torch.mean(attr_t)) / torch.std(attr_t)
        else:
            tmp = (attr_t - attr_t.min()) / (attr_t.max() - attr_t.min())
    if is_tuple:
        return tuple(tmp)
    return tmp

@register_evaluator(EvaluatorKeys.CONTINUITY)
class ContinuityEvaluator(EvaluatorBase):
    def __init__(
        self,
        output_file: str,
        summary_writer: SummaryWriter,
        attr_map_key: str,
        overwrite: bool = False,
    ):
        super().__init__(
            output_file=output_file, summary_writer=summary_writer, overwrite=overwrite
        )

        self._attr_map_key = attr_map_key

    def compute(self, engine: Engine):
        attr_map = getattr(engine.state, self._attr_map_key)

        # for continuity we normalize the attributions from -1 to 1
        attr_map = attr_map / attr_map.abs().max()

        # generate continuity directly
        grad_y = (
            torch.abs(attr_map[:, :, 1:, :] - attr_map[:, :, :-1, :])
            .mean(axis=(1, 2, 3))
            .view(-1, 1)
        )
        grad_x = (
            torch.abs(attr_map[:, :, :, 1:] - attr_map[:, :, :, :-1])
            .mean(axis=(1, 2, 3))
            .view(-1, 1)
        )
        continuity = torch.cat([grad_x, grad_y], axis=1).mean(axis=1)

        # get the continuity scores
        return continuity.detach().cpu().numpy()

    def fire_completion_event(self, engine: Engine):
        engine.fire_event(EvaluatorEvents.CONTINUITY_COMPUTED)

from torch.nn.functional import softmax
def wrap_model_output(torch_model_unwrapped: torch.nn.Module):
    class Wrapped(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self._model = model

        def forward(self, *args, **kwargs):
            output = softmax(self._model(*args, **kwargs))
            # print('output', output)
            return output

        def __getattr__(self, name):
            """Forward missing attributes to twice-wrapped module."""
            try:
                # defer to nn.Module's logic
                return super().__getattr__(name)
            except AttributeError:
                # forward to the twice-wrapped module
                return getattr(self._model, name)

    return Wrapped(torch_model_unwrapped)

@register_evaluator(EvaluatorKeys.INFIDELITY)
class InfidelityEvaluator(EvaluatorBase):
    def __init__(
        self,
        output_file: str,
        summary_writer: SummaryWriter,
        model: torch.nn.Module,
        attr_map_key: str,
        perturbation_noise: float = 0.1,
        n_perturb_samples: int = 10,
        normalize: bool = True,
        overwrite: bool = False,
    ):
        super().__init__(
            output_file=output_file, summary_writer=summary_writer, overwrite=overwrite
        )

        # self._model = wrap_model_output(model)
        self._model = model
        self._attr_map_key = attr_map_key
        self._perturbation_noise = perturbation_noise
        self._normalize = normalize
        self._n_perturb_samples = n_perturb_samples
        self._output_transform = lambda x: (x[DataKeys.IMAGE], x[DataKeys.PRED])

    def compute(self, engine: Engine) -> None:
        attr_map = getattr(engine.state, self._attr_map_key)

        # for infidelity we normalize the attributions from -1 to 1
        # attr_map = attr_map / attr_map.abs().max()

        def perturb_fn(inputs):
            noise = torch.tensor(
                np.random.normal(0, self._perturbation_noise, inputs.shape),
                device=inputs.get_device(),
            ).float()
            return noise, inputs - noise

        # get images and preds
        image, pred = self._output_transform(engine.state.output)

        # get the attribution map from captum
        print('attr_map', attr_map.min(), attr_map.max())
        print("Computing infidelity with: ", 'batch size:', image.shape[0], 'n_perturb_samples: ', self._n_perturb_samples, 'perturbation_noise: ', self._perturbation_noise, 'normalize: ', self._normalize)
        output = infidelity(
                self._model,
                perturb_fn,
                image,
                attr_map.cuda(),
                target=pred,
                max_examples_per_batch=image.shape[0],
                n_perturb_samples=self._n_perturb_samples,
                normalize=self._normalize,
            ).detach().cpu()
        print("Infidelity = ", output, output.mean())
        return output

    def fire_completion_event(self, engine: Engine):
        engine.fire_event(EvaluatorEvents.INFIDELITY_COMPUTED)


@register_evaluator(EvaluatorKeys.SENSITIVITY)
class SensitivityEvaluator(EvaluatorBase):
    def __init__(
        self,
        output_file: str,
        summary_writer: SummaryWriter,
        attr: CaptumAttrWrapperBase,
        baselines: torch.Tensor,
        perturb_radius: float = 0.02,
        n_perturb_samples: int = 10,
        overwrite: bool = False,
        output_transform=lambda x: {
            key: val for key, val in x.items() if key in [DataKeys.IMAGE, DataKeys.PRED]
        },
    ):
        super().__init__(
            output_file=output_file, summary_writer=summary_writer, overwrite=overwrite
        )

        self._attr = attr
        self._baselines = baselines
        self._perturb_radius = perturb_radius
        self._n_perturb_samples = n_perturb_samples
        self._output_transform = output_transform

    def compute(self, engine: Engine) -> None:
        inputs = self._output_transform(engine.state.output)

        if "image" in inputs:
            inputs["inputs"] = inputs["image"]
            del inputs["image"]
        if "pred" in inputs:
            inputs["target"] = inputs["pred"]
            del inputs["pred"]

        print('baseline', ('baseline' in inputs))
        print("Computing sensitivity with: ", 'batch size:', inputs['inputs'].shape[0], 'n_perturb_samples: ', self._n_perturb_samples, 'perturb_radius: ', self._perturb_radius)
        if "baseline" in inputs:
            return (
                sensitivity_max(
                    self._attr.attribute,
                    **inputs,
                    perturb_radius=self._perturb_radius,
                    n_perturb_samples=self._n_perturb_samples,
                    # this must always remain equal to the batch size, otherwise sensitivity will not be correctly computed!!
                    # this allows internal sensitivty implementation to simply perturb each batch and get corresponding attributions
                    # for senstivity. The max is then taken across n_perturb_samples to get the sensitivity score
                    max_examples_per_batch=inputs['inputs'].shape[0],
                )
                .detach()
                .cpu()
            )
        else:
            return (
                sensitivity_max(
                    self._attr.attribute,
                    **inputs,
                    baselines=self._baselines,
                    perturb_radius=self._perturb_radius,
                    n_perturb_samples=self._n_perturb_samples,
                    # this must always remain equal to the batch size, otherwise sensitivity will not be correctly computed!!
                    # this allows internal sensitivty implementation to simply perturb each batch and get corresponding attributions
                    # for senstivity. The max is then taken across n_perturb_samples to get the sensitivity score
                    max_examples_per_batch=inputs['inputs'].shape[0],
                )
                .detach()
                .cpu()
            )

    def fire_completion_event(self, engine: Engine):
        engine.fire_event(EvaluatorEvents.SENSITIVITY_COMPUTED)


    def fire_completion_event(self, engine: Engine):
        engine.fire_event(EvaluatorEvents.SENSITIVITY_COMPUTED)


@register_evaluator(EvaluatorKeys.FEATURE_PERTURBATION)
class FeaturePerturbationEvaluator(EvaluatorBase):
    def __init__(
        self,
        output_file: str,
        summary_writer: SummaryWriter,
        model: torch.nn.Module,
        attr_map_key: str,
        feature_perturber: FeaturePerturber,
        visualize: bool = False,
        overwrite: bool = False,
        n_random_runs: int = 10,
        is_random_run: bool = False,
        importance_orders: List[str] = ["ascending", "descending"],
    ):
        super().__init__(
            output_file=output_file, summary_writer=summary_writer, overwrite=overwrite
        )
        self._model = model
        self._attr_map_key = attr_map_key
        self._feature_perturber = feature_perturber
        self._visualize = visualize
        self._n_random_runs = n_random_runs
        self._is_random_run = is_random_run
        self._importance_orders = importance_orders
        self._output_transform = lambda x: (
            x[DataKeys.IMAGE],
            x[DataKeys.PRED],
            x[DataKeys.ORIG_IMAGE],
        )

    def transform_data_on_load(self, data):
        return data

    def finished(self, name, max_data_required: int):
        if self._overwrite:
            return False
        # read from h5 if this data is already computed
        if Path(self._output_file).exists():
            hf = h5py.File(self._output_file, "r")

            # check if required data is already computed
            if name in hf:
                for order, data in hf[name].items():
                    data = [dict(zip(data, t)) for t in zip(*data.values())]
                    if len(data) == max_data_required:
                        return True
            return False

    def read_data_from_hdf5(self, engine: Engine, name: str, output_file: str):
        # read from h5 if this data is already computed
        if Path(output_file).exists():
            hf = h5py.File(output_file, "r")

            # get data indices
            indices = np.array(engine.state.batch["index"])

            if name in hf:
                output_data = {}
                base_data = hf[name]

                if not self._is_random_run:
                    for order in self._importance_orders:
                        if order not in base_data.keys():
                            return False
                else:
                    import tqdm
                    for idx in tqdm.tqdm(range(self._n_random_runs)):
                        if f"random_{idx}" not in base_data:
                            return False

                for order, data in base_data.items():
                    data = [dict(zip(data, t)) for t in zip(*data.values())]

                    # check if required data is already computed
                    if indices.max() < len(data):
                        indexed_data = [data[idx] for idx in indices]
                        output_data[order] = indexed_data
                if len(output_data) > 0:
                    return self.transform_data_on_load(output_data)

    def write_data_to_hdf5(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.array
    ):
        # add labels
        data = getattr(engine.state, EvaluatorKeys.FEATURE_PERTURBATION)
        if EvaluatorKeys.FEATURE_PERTURBATION not in hf:
            base_group = hf.create_group(EvaluatorKeys.FEATURE_PERTURBATION)
        else:
            base_group = hf[EvaluatorKeys.FEATURE_PERTURBATION]
        for order, results in data.items():
            if order not in base_group:
                base_group.create_group(order)
            results = {k: [dic[k] for dic in results] for k in results[0]}
            for key, data in results.items():
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                elif isinstance(data, list):
                    data = np.array(data)
                update_dataset_at_indices(
                    base_group[order],
                    key=key,
                    indices=indices,
                    data=data,
                    maxshape=(None, *data.shape[1:]),
                    overwrite=self._overwrite,
                )

    def compute(self, engine: Engine) -> None:
        # generate perturbation results for removal of top features, bottom features,
        # and random features
        results = {}
        image, pred, original_image = self._output_transform(engine.state.output)
        if not self._is_random_run:
            attr_map = getattr(engine.state, self._attr_map_key)
            for order in self._importance_orders:
                print("order", order)
                perturbation_results = self._feature_perturber.run(
                    self._model,
                    inputs=original_image,
                    targets=pred,  # todo: allow target to be both pred or original sample label
                    attr_maps=attr_map,
                    importance_order=order,
                )
                results[order] = perturbation_results
        else:
            import tqdm
            for order in tqdm.tqdm(range(self._n_random_runs)):
                perturbation_results = self._feature_perturber.run(
                    self._model,
                    inputs=original_image,
                    targets=pred,  # todo: allow target to be both pred or original sample label
                    attr_maps=torch.ones_like(image),
                    importance_order="random",
                )
                results[f"random_{order}"] = perturbation_results

        return results

    def fire_completion_event(self, engine: Engine):
        engine.fire_event(EvaluatorEvents.FEATURE_PERTURBATION_COMPUTED)

@register_evaluator(EvaluatorKeys.AOPC)
class AOPCEvaluator(EvaluatorBase):
    def __init__(
        self,
        output_file: str,
        summary_writer: SummaryWriter,
        overwrite: bool = False,
    ):
        super().__init__(
            output_file=output_file, summary_writer=summary_writer, overwrite=overwrite
        )
        self.total_score = 0
        self.total_counts = 0

    def compute(self, engine: Engine) -> None:
        perturbation_results = getattr(engine.state, EvaluatorKeys.FEATURE_PERTURBATION)
        per_order_scores = {}
        for order, perturbation_results_per_sample in perturbation_results.items():
            if order not in per_order_scores:
                per_order_scores[order] = []
            for perturbation_result in perturbation_results_per_sample:
                cumulative_value = 0.0
                aopc_scores_per_sample = []
                conf_scores = np.fromstring(
                    perturbation_result["conf_score"], dtype=np.float32
                )
                for n, curr_conf_score in enumerate(conf_scores):
                    cumulative_value += conf_scores[0] - curr_conf_score
                    aopc_scores_per_sample.append(cumulative_value / (n + 1))
                per_order_scores[order].append(aopc_scores_per_sample)
            per_order_scores[order] = np.array(per_order_scores[order])

        if 'descending' in per_order_scores:
            from scipy.integrate import simps
            n_steps = per_order_scores['descending'].shape[1]
            morf = np.array(per_order_scores['descending']).mean(0)
            lerf = np.array(per_order_scores['ascending']).mean(0)

            morf_mean = simps(morf, np.arange(0, n_steps))
            lerf_mean = simps(lerf, np.arange(0, n_steps))

            bs = np.array(per_order_scores['descending']).shape[0]
            self.total_score += (morf_mean - lerf_mean) * bs
            self.total_counts += bs
            # print('total_score', self.total_score * np.array(per_order_scores['descending']).shape[0])
            # print("total_counts",  self.total_counts)
            print("score = ", self.total_score / self.total_counts)

        return per_order_scores

    def finished(self, name, max_data_required: int):
        # read from h5 if this data is already computed
        if Path(self._output_file).exists():
            hf = h5py.File(self._output_file, "r")

            # check if required data is already computed
            if name in hf:
                for order, data in hf[name].items():
                    if len(data) == max_data_required:
                        return True
            return False

    def read_data_from_hdf5(self, engine: Engine, name: str, output_file: str):
        # read from h5 if this data is already computed
        if Path(output_file).exists():
            hf = h5py.File(output_file, "r")

            # get data indices
            indices = np.array(engine.state.batch["index"])

            if name in hf:
                output_data = {}
                base_data = hf[name]
                for order, data in base_data.items():
                    # check if required data is already computed
                    if indices.max() < len(data):
                        output_data[order] = data[indices]
                if len(output_data) > 0:
                    return self.transform_data_on_load(output_data)

    def write_data_to_hdf5(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.array
    ):
        # add labels
        data = getattr(engine.state, EvaluatorKeys.AOPC)
        if EvaluatorKeys.AOPC not in hf:
            base_group = hf.create_group(EvaluatorKeys.AOPC)
        else:
            base_group = hf[EvaluatorKeys.AOPC]
        for order, data in data.items():
            update_dataset_at_indices(
                base_group,
                key=order,
                indices=indices,
                data=data,
                maxshape=(None, *data.shape[1:]),
                overwrite=self._overwrite,
            )

    def fire_completion_event(self, engine: Engine):
        engine.fire_event(EvaluatorEvents.AOPC_COMPUTED)



class DataSaverHandler:
    def __init__(self, output_file, attached_evaluators: Mapping[str, EvaluatorBase]):
        self._attached_evaluators = attached_evaluators
        self._output_file = output_file

    def add_key_from_output(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.array
    ):
        # add labels
        data = engine.state.output[key]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        max_shape = (None,)
        if isinstance(data, np.ndarray):
            max_shape = (None, *data.shape[1:])
        elif isinstance(data, list):
            max_shape = (None,)
        update_dataset_at_indices(
            hf, key=key, indices=indices, data=data, maxshape=max_shape, overwrite=True
        )

    def __call__(self, engine: Engine) -> None:
        # this part is quite slow? maybe we can speed up by not overwriting but just leaving already written indces?
        if not Path(self._output_file).parent.exists():
            Path(self._output_file).parent.mkdir(parents=True)

        hf = h5py.File(self._output_file, "a")

        # get data indices
        indices = engine.state.batch["index"]
        indices = np.array(indices)
        print('indices', indices)

        # create index dataset
        update_dataset_at_indices(hf, key="index", indices=indices, data=indices)

        for key, evaluator in self._attached_evaluators.items():
            evaluator.write_data_to_hdf5(engine, hf, key, indices)

        # add labels
        self.add_key_from_output(engine, hf, DataKeys.LABEL, indices)

        # add preds
        self.add_key_from_output(engine, hf, DataKeys.PRED, indices)

        # add image_file paths
        self.add_key_from_output(engine, hf, DataKeys.IMAGE_FILE_PATH, indices)

        hf.close()
