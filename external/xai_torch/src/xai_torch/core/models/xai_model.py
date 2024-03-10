""" PyTorch module that defines the base model for training/testing etc. """


from __future__ import annotations

import logging
from typing import IO, TYPE_CHECKING, Any, Dict, Optional, Union

from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from xai_torch.core.models.utilities.checkpoints import filter_keys, load
from xai_torch.core.training.constants import TrainingStage
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    from ignite.engine import Engine

    from xai_torch.core.args import Arguments
    from xai_torch.core.models.base import BaseModule


# todo: Update this according to ignite checkpoint loading
class XAIModelIO:
    @classmethod
    def load_from_checkpoint(
        cls,
        model: XAIModel,
        checkpoint_path: Union[str, IO],
        strict: bool = True,
    ):
        """
        We remove hparams loading functionality from checkpoint loading unlike
        lightning.
        """
        import ignite.distributed as idist

        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        if idist.get_world_size() > 1:
            checkpoint = load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = load(checkpoint_path, map_location=idist.device())

        keys = model.on_load_checkpoint(checkpoint, strict=strict)
        if not strict:
            if keys.missing_keys:
                logger.warning(
                    f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
                )
            if keys.unexpected_keys:
                logger.warning(
                    f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
                )

        return model


class XAIModel(XAIModelIO):
    def __init__(
        self,
        args: Arguments,
        model_class: BaseModule,
        tb_logger: Optional[TensorboardLogger] = None,
        gt_metadata: Optional[dict] = None,
    ):
        super().__init__()

        import ignite.distributed as idist

        # initialize arguments
        self._args = args
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        self._tb_logger = tb_logger
        self._ema_handler = None
        self.activate_ema = False

        # models sometimes download pretrained checkpoints when initializing. Only download it on rank 0
        if idist.get_rank() > 0:  # stop all ranks > 0
            idist.barrier()

        if args.model_args.model_task == "image_classification":
            if "class_labels" not in gt_metadata:
                raise ValueError("class_labels are required in gt_metadata for image_classification tasks.")
            self._torch_model = model_class(
                args,
                class_labels=gt_metadata["class_labels"],
            )
        else:
            self._torch_model = model_class(
                args,
            )

        # build model
        self._torch_model.build_model()

        # initialize metrics
        self._torch_model.init_metrics()

        # wait for rank 0 to download checkpoints
        if idist.get_rank() == 0:
            idist.barrier()

    def setup(self, stage: TrainingStage):
        import ignite.distributed as idist
        from torch.nn.parallel import DataParallel

        from xai_torch.core.models.ddp_model_proxy import ModuleProxyWrapper
        from xai_torch.core.models.utilities.general import batch_norm_to_group_norm

        # adapt model for distributed settings if configured and put it to device
        self.model_to_device()

        # wrap the ddp module with the proxy wrapper to forward missing attr requests to the underlying model
        if idist.get_world_size() > 1 and stage == TrainingStage.train:
            self._torch_model = ModuleProxyWrapper(self._torch_model)
        elif isinstance(self._torch_model, DataParallel):
            self._torch_model = ModuleProxyWrapper(self._torch_model)

        # replace batch norm with group norm if required
        if self._args.model_args.convert_bn_to_gn:
            batch_norm_to_group_norm(self._torch_model)

        # add model ema if required
        from ignite.handlers.ema_handler import EMAHandler

        if self._args.training_args.model_ema_args.enabled:
            self._ema_handler = EMAHandler(
                self._torch_model,
                momentum=self._args.training_args.model_ema_args.momentum,
                momentum_warmup=self._args.training_args.model_ema_args.momentum_warmup,
                warmup_iters=self._args.training_args.model_ema_args.warmup_iters,
            )
            self._ema_model = self._ema_handler.ema_model

    def model_to_device(self, device="gpu"):
        import ignite.distributed as idist
        import torch

        device = idist.device() if device == "gpu" else device
        self._torch_model = idist.auto_model(
            self._torch_model,
            sync_bn=False if device == torch.device("cpu") else self._args.training_args.sync_batchnorm,
        )

    def add_training_hooks(self, training_engine: Engine):
        self.torch_model.add_training_hooks(training_engine)

    @property
    def ema_handler(self):
        return self._ema_handler

    @property
    def model_name(self):
        return self._torch_model.model_name

    @property
    def torch_model(self):
        if self.activate_ema:
            return self._ema_model
        else:
            return self._torch_model

    def summarize(self):
        from torchinfo import summary

        from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

        logger = logging.getLogger(DEFAULT_LOGGER_NAME)

        logger.info(summary(self.torch_model, verbose=0))

    def get_checkpoint_map(self, checkpoint: Dict[str, Any]) -> None:
        """
        Called from checkpoint connector when saving checkpoints
        """

        # add model to checkpoint
        checkpoint["model"] = self._torch_model

        # if ema model is available, save it
        if self._ema_handler is not None:
            checkpoint["ema_model"] = self._ema_handler.ema_model

        # add additional information from the underlying model if needed
        self._torch_model.get_checkpoint_map(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any], strict: bool = True):
        state_dict_key = self._args.model_args.checkpoint_state_dict_key
        if state_dict_key not in checkpoint:  # maybe we are just resuming?
            self._logger.warning(
                f"Requested state dict key [{state_dict_key}] does not exist "
                "in the model checkpoint. Using key 'state_dict'."
            )
            state_dict_key = "state_dict"
        else:
            if state_dict_key != "state_dict":
                checkpoint["state_dict"] = checkpoint[state_dict_key]
                checkpoint.pop(state_dict_key)
                state_dict_key = "state_dict"

        checkpoint = filter_keys(checkpoint, state_dict_key, keys=["_wrapped_model."])
        return self._torch_model.on_load_checkpoint(checkpoint, strict=strict)
