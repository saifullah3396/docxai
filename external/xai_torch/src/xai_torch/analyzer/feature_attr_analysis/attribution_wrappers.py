"""
Defines the feature attribution generation task.
"""

import typing
from abc import abstractproperty
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
import torch
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    GuidedBackprop,
    GuidedGradCam,
    InputXGradient,
    IntegratedGradients,
    KernelShap,
    Lime,
    NoiseTunnel,
    Occlusion,
    Saliency,
)
from captum.attr._core.deep_lift import SUPPORTED_NON_LINEAR, nonlinear
from lime.wrappers.scikit_image import SegmentationAlgorithm
from shap.explainers._deep.deep_pytorch import nonlinear_1d, op_handler
from torch import nn
from xai_torch.utilities.abstract_dataclass import AbstractDataclass
from xai_torch.utilities.decorators import register_as_child
import tqdm

# add GELU to DeepLIFT and DeepSHAP
SUPPORTED_NON_LINEAR[nn.GELU] = nonlinear
op_handler["GELU"] = nonlinear_1d


class GridSegmenter:
    def __init__(self, cell_size: int):
        self._cell_size = cell_size

    def __call__(self, arr: np.ndarray):
        h, w, _ = arr.shape
        feature_mask = np.arange(h // self._cell_size * w // self._cell_size).reshape(
            h // self._cell_size, w // self._cell_size
        )
        return np.kron(feature_mask, np.ones((self._cell_size, self._cell_size))).astype(int)


ATTRIBUTIONS_REGISTRY = {}


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


def register_attribution(reg_name: str = ""):
    return register_as_child(
        base_class_type=AbstractDataclass,
        registry=ATTRIBUTIONS_REGISTRY,
        reg_name=reg_name,
    )


@dataclass
class AttrWrapperBase(AbstractDataclass):
    model: torch.nn.Module
    batch_size: int = 8


@dataclass
class CaptumAttrWrapperBase(AttrWrapperBase):
    apply_baseline: bool = False
    apply_noise_tunnel: bool = False
    nt_samples: int = 10
    nt_type: str = "smoothgrad"
    normalize: bool = False
    per_sample: bool = True

    @abstractproperty
    def attr_cls(self):
        pass

    def __post_init__(self) -> None:
        self._attr = self.attr_cls(self.model)
        if self.apply_noise_tunnel:
            self._attr = NoiseTunnel(self._attr)

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        if isinstance(self._attr, NoiseTunnel):
            attr = self._attr.attribute(
                inputs=inputs, target=target, baselines=baselines, nt_samples=self.nt_samples, nt_type=self.nt_type
            )
        else:
            if self.apply_baseline:
                attr = self._attr.attribute(
                    inputs=inputs,
                    target=target,
                    baselines=baselines,
                )
            else:
                attr = self._attr.attribute(
                    inputs=inputs,
                    target=target,
                )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("integrated_gradients")
@dataclass
class IntegratedGradientsWrapper(CaptumAttrWrapperBase):
    n_steps: int = 50
    method: str = "gausslegendre"
    internal_batch_size: Optional[int] = 32

    @property
    def attr_cls(self):
        return IntegratedGradients

    def __post_init__(self) -> None:
        self._attr = self.attr_cls(self.model, multiply_by_inputs=False)

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        if isinstance(self._attr, NoiseTunnel):
            attr = super().attribute(inputs=inputs, target=target, baselines=baselines)
        if not self.apply_baseline:
            baselines = None
        attr = self._attr.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            n_steps=self.n_steps,
            method=self.method,
            internal_batch_size=self.internal_batch_size,
        )

        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)

        return attr


@register_attribution("saliency")
@dataclass
class SaliencyWrapper(CaptumAttrWrapperBase):
    @property
    def attr_cls(self):
        return Saliency

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        nt_kwargs = {}
        if isinstance(self._attr, NoiseTunnel):
            nt_kwargs = dict(nt_samples=self.nt_samples, nt_type=self.nt_type)
        if isinstance(inputs, tuple):
            print('Saliency', inputs[0].shape, target[0].shape, baselines[0].shape)
        else:
            print('Saliency', inputs.shape, target.shape, baselines.shape)
        attr = self._attr.attribute(
            inputs=inputs,
            target=target,
            abs=False,
            **nt_kwargs
        )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("deeplift")
@dataclass
class DeepLiftWrapper(CaptumAttrWrapperBase):
    @property
    def attr_cls(self):
        return DeepLift

    def __post_init__(self) -> None:
        self._attr = self.attr_cls(self.model, multiply_by_inputs=False)

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        if isinstance(self._attr, NoiseTunnel):
            attr = self._attr.attribute(
                inputs=inputs, target=target, baselines=baselines, nt_samples=self.nt_samples, nt_type=self.nt_type
            )
        else:
            if self.apply_baseline:
                baselines = baselines[0]
            else:
                baselines = None
            attr = self._attr.attribute(inputs=inputs, target=target, baselines=None)
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("deeplift_shap")
@dataclass
class DeepLiftShapWrapper(CaptumAttrWrapperBase):
    @property
    def attr_cls(self):
        return DeepLiftShap

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        if isinstance(self._attr, NoiseTunnel):
            attr = self._attr.attribute(
                inputs=inputs, target=target, baselines=baselines, nt_samples=self.nt_samples, nt_type=self.nt_type
            )
        else:
            attr = self._attr.attribute(
                inputs=inputs,
                target=target,
                baselines=baselines,
            )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("deepshap")
@dataclass
class DeepShap(AttrWrapperBase):
    background_samples: torch.Tensor = None
    internal_batch_size: int = 10
    normalize: bool = False
    per_sample: bool = True

    def __post_init__(self) -> None:
        import shap

        if self.background_samples is None:
            raise AttributeError("background_samples tensor must be provided at initialization.")

        self._explainer = shap.DeepExplainer(
            self.model, self.background_samples, internal_batch_size=self.internal_batch_size
        )

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        # Important Note!!!:
        # when comptuing senstivities, inputs is a single perturbed batch only if max_examples_per_batch is set equal to the batch size
        # this implementation only works if each batch is perturbed indepdenently, therefore inputs shape and targets shape must be the same
        # basically, this means that attribution for each batch is just computed n_perburb_samples times separately
        # therefore a for loop over the batch works with target[idx]
        # otherwise, the the repeated inputs may not match the corresponding targets

        return_tuple = False
        if isinstance(inputs, tuple):  # this is for sensitivity/infidelity
            inputs = inputs[0]
            return_tuple = True
        shap_values_list = []

        assert inputs.shape[0] == target.shape[0], "inputs and target must have the same batch size"
        for idx, input in tqdm.tqdm(enumerate(inputs)):
            shap_values = self._explainer.shap_values(
                input.unsqueeze(0), output_idx=target[idx], check_additivity=False
            )[0]
            shap_values_list.append(torch.from_numpy(shap_values))
        attr = torch.cat(shap_values_list)
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)

        if return_tuple:
            return (attr,)
        else:
            return attr


@register_attribution("gradient_shap")
@dataclass
class GradientShapWrapper(CaptumAttrWrapperBase):
    n_samples: int = 5
    stdevs: Union[float, Tuple[float, ...]] = 0
    n_baselines: int = 20

    @property
    def attr_cls(self):
        return GradientShap

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        if baselines is None:
            baselines = torch.randn(self.n_baselines, *inputs.shape[1:])
        attr = self._attr.attribute(
            inputs=inputs, target=target, baselines=baselines, n_samples=self.n_samples, stdevs=float(self.stdevs)
        )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("input_x_gradient")
@dataclass
class InputXGradientWrapper(CaptumAttrWrapperBase):
    @property
    def attr_cls(self):
        return InputXGradient

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        attr = self._attr.attribute(
            inputs=inputs,
            target=target,
        )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("guided_backprop")
@dataclass
class GuidedBackpropWrapper(CaptumAttrWrapperBase):
    @property
    def attr_cls(self):
        return GuidedBackprop

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        attr = self._attr.attribute(
            inputs=inputs,
            target=target,
        )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("guided_grad_cam")
@dataclass
class GuidedGradCamWrapper(CaptumAttrWrapperBase):
    @property
    def attr_cls(self):
        return GuidedGradCam


@register_attribution("feature_ablation")
@dataclass
class FeatureAblationWrapper(CaptumAttrWrapperBase):
    strides: int = 3
    sliding_window_shapes: int = 5
    baseline_value: int = 0
    normalize: bool = False
    per_sample: bool = True

    @property
    def attr_cls(self):
        return FeatureAblation

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        self.baseline_value = torch.mean(inputs, dim=(0, 2))
        baselines = self.baseline_value.unsqueeze(1).repeat_interleave(inputs[0].shape[1], dim=1)
        baselines = baselines.reshape(self.baseline_value.shape[0], inputs[0].shape[1])
        if isinstance(inputs, tuple):
            stride = inputs[0].shape[1]
        else:
            stride = inputs.shape[1]
        attr = self._attr.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            strides=(stride, self.strides),
            sliding_window_shapes=(stride, self.sliding_window_shapes),
        )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("occlusion")
@dataclass
class OcclusionWrapper(CaptumAttrWrapperBase):
    baseline_value: int = 0
    strides: typing.Any = (3, 8, 8)
    sliding_window_shapes: typing.Any = (3, 15, 15)
    perturbations_per_eval: int =100

    @property
    def attr_cls(self):
        return Occlusion

    def __post_init__(self) -> None:
        super().__post_init__()

        if isinstance(self.strides, list):
            self.strides = tuple(self.strides)

        if isinstance(self.sliding_window_shapes, list):
            self.sliding_window_shapes = tuple(self.sliding_window_shapes)

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        print(f"Computing attrs for batch...")
        attr = self._attr.attribute(
            inputs=inputs,
            target=target,
            baselines=self.baseline_value,
            strides=self.strides,
            sliding_window_shapes=self.sliding_window_shapes,
            perturbations_per_eval=self.perturbations_per_eval,
            show_progress=True,
        )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("feature_permutation")
@dataclass
class FeaturePermutationWrapper(CaptumAttrWrapperBase):
    strides: int = 3
    sliding_window_shapes: int = 5
    normalize: bool = False
    per_sample: bool = True

    @property
    def attr_cls(self):
        return FeaturePermutation

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        if isinstance(inputs, tuple):
            stride = inputs[0].shape[1]
        else:
            stride = inputs.shape[1]
        attr = self._attr.attribute(
            inputs=inputs,
            target=target,
            strides=(stride, self.strides),
            sliding_window_shapes=(stride, self.sliding_window_shapes),
        )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("ts_shapley_value_sampling")
@dataclass
class TSShapleyValueSamplingWrapper(CaptumAttrWrapperBase):
    n_samples: int = 200
    cell_size: int = 5
    normalize: bool = False
    per_sample: bool = True

    @property
    def attr_cls(self):
        return ShapleyValueSampling

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        # if feature_mask is not provided we generate it here with lime original segmentation algorithms
        if feature_masks is None:
            segmenter = TSGridSegmenter(self.cell_size)
            feature_masks = []
            for input in inputs:
                feature_masks.append(torch.from_numpy(segmenter(input.permute(1, 0).cpu().numpy())))
            feature_masks = torch.stack(feature_masks).to(inputs.get_device())

        attr = self._attr.attribute(
            inputs=inputs,
            target=target,
            n_samples=self.n_samples,
            feature_mask=feature_masks,
        )
        if self.normalize:
            return normalize(attr, per_sample=self.per_sample)
        return attr


@register_attribution("lime")
@dataclass
class LimeShapWrapper(CaptumAttrWrapperBase):
    n_samples: int = 200
    perturbations_per_eval: int = 200
    segment_fn: str = "slic"
    segment_fn_kwargs: dict = field(default_factory=lambda: {"n_segments": 100, "compactness": 1, "sigma": 1})
    # segment_fn: str = "grid"
    # segment_fn_kwargs: dict = field(default_factory=lambda: {"cell_size": 32})

    @property
    def attr_cls(self):
        return Lime

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        return_tuple = False
        if isinstance(inputs, tuple):  # this is for sensitivity/infidelity
            inputs = inputs[0]
            return_tuple = True

        # if feature_mask is not provided we generate it here with lime original segmentation algorithms
        if feature_masks is None:
            if self.segment_fn == "grid":
                segmenter = GridSegmenter(**self.segment_fn_kwargs)
            else:
                segmenter = SegmentationAlgorithm(self.segment_fn, **self.segment_fn_kwargs)
            feature_masks = []
            print("Generating feature masks...")
            for input in tqdm.tqdm(inputs):
                feature_masks.append(torch.from_numpy(segmenter(input.permute(1, 2, 0).cpu().numpy())))
            feature_masks = torch.stack(feature_masks).to(inputs.get_device()) - 1  # minus 1 to start indices from 0

        attributions = []
        print("Generating attributions...")
        for idx, input in tqdm.tqdm(enumerate(inputs)):
            attribution = self._attr.attribute(
                inputs=input.unsqueeze(0),
                target=target[idx],
                n_samples=self.n_samples,
                feature_mask=feature_masks[idx],
                perturbations_per_eval=self.perturbations_per_eval,
            )
            attributions.append(attribution)
        attributions = torch.cat(attributions)

        if self.normalize:
            attributions = normalize(attributions, per_sample=self.per_sample)

        if return_tuple:
            return (attributions,)
        else:
            return attributions


@register_attribution("ts_lime")
@dataclass
class TSLimeShapWrapper(CaptumAttrWrapperBase):
    n_samples: int = 200
    cell_size: int = 5
    normalize: bool = False
    per_sample: bool = True

    @property
    def attr_cls(self):
        return Lime

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        return_tuple = False
        if isinstance(inputs, tuple):  # this is for sensitivity/infidelity
            inputs = inputs[0]
            return_tuple = True

        # if feature_mask is not provided we generate it here with lime original segmentation algorithms
        if feature_masks is None:
            segmenter = TSGridSegmenter(self.cell_size)
            feature_masks = []
            for input in inputs:
                feature_masks.append(torch.from_numpy(segmenter(input.permute(1, 0).cpu().numpy())))
            feature_masks = torch.stack(feature_masks).to(inputs.get_device())

        attributions = []
        for idx, input in enumerate(inputs):
            attribution = self._attr.attribute(
                inputs=input.unsqueeze(0),
                target=target[idx],
                n_samples=self.n_samples,
                feature_mask=feature_masks[idx],
            )
            attributions.append(attribution)
        attributions = torch.cat(attributions)
        if self.normalize:
            attributions = normalize(attributions, per_sample=self.per_sample)
        if return_tuple:
            return (attributions,)
        else:
            return attributions


@register_attribution("kernel_shap")
@dataclass
class KernelShapWrapper(CaptumAttrWrapperBase):
    n_samples: int = 200
    perturbations_per_eval: int = 200
    segment_fn: str = "slic"
    segment_fn_kwargs: dict = field(default_factory=lambda: {"n_segments": 100, "compactness": 1, "sigma": 1})

    @property
    def attr_cls(self):
        return KernelShap

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        return_tuple = False
        if isinstance(inputs, tuple):  # this is for sensitivity/infidelity
            inputs = inputs[0]
            return_tuple = True

        # if feature_mask is not provided we generate it here with lime original segmentation algorithms
        if feature_masks is None:
            if self.segment_fn == "grid":
                segmenter = GridSegmenter(**self.segment_fn_kwargs)
            else:
                segmenter = SegmentationAlgorithm(self.segment_fn, **self.segment_fn_kwargs)
            feature_masks = []
            print("Generating feature masks...")
            for input in tqdm.tqdm(inputs):
                feature_masks.append(torch.from_numpy(segmenter(input.permute(1, 2, 0).cpu().numpy())))
            feature_masks = torch.stack(feature_masks).to(inputs.get_device()) - 1  # minus 1 to start indices from 0

        attributions = []
        print("Generating attributions...")
        for idx, input in tqdm.tqdm(enumerate(inputs)):
            try:
                attribution = self._attr.attribute(
                    inputs=input.unsqueeze(0),
                    target=target[idx],
                    n_samples=self.n_samples,
                    feature_mask=feature_masks[idx],
                    perturbations_per_eval=self.perturbations_per_eval
                )
                attributions.append(attribution)
            except Exception as e:
                attributions.append(torch.zeros(input.unsqueeze(0).shape, device=input.get_device()))
        attributions = torch.cat(attributions)

        if self.normalize:
            attributions = normalize(attributions, per_sample=self.per_sample)

        if return_tuple:
            return (attributions,)
        else:
            return attributions


@register_attribution("ts_kernel_shap")
@dataclass
class TSKernelShapWrapper(CaptumAttrWrapperBase):
    n_samples: int = 200
    cell_size: int = 5
    normalize: bool = False
    per_sample: bool = True

    @property
    def attr_cls(self):
        return KernelShap

    def attribute(self, inputs, target, baselines=None, feature_masks=None):
        return_tuple = False
        if isinstance(inputs, tuple):  # this is for sensitivity/infidelity
            inputs = inputs[0]
            return_tuple = True

        # if feature_mask is not provided we generate it here with lime original segmentation algorithms
        if feature_masks is None:
            segmenter = TSGridSegmenter(self.cell_size)
            feature_masks = []
            for input in inputs:
                feature_masks.append(torch.from_numpy(segmenter(input.permute(1, 0).cpu().numpy())))
            feature_masks = torch.stack(feature_masks).to(inputs.get_device())

        attributions = []
        for idx, input in enumerate(inputs):
            attribution = self._attr.attribute(
                inputs=input.unsqueeze(0),
                target=target[idx],
                n_samples=self.n_samples,
                feature_mask=feature_masks[idx],
            )
            attributions.append(attribution)
        attributions = torch.cat(attributions)
        if self.normalize:
            attributions = normalize(attributions, per_sample=self.per_sample)
        if return_tuple:
            return (attributions,)
        else:
            return attributions
