"""Helper functions for the tiled ensemble training."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from lightning import Trainer

from anomalib.data import AnomalibDataModule, get_datamodule
from anomalib.models import AnomalyModule, get_model
from anomalib.pipelines.tiled_ensemble.normalization import NormalizationStage
from anomalib.utils.normalization import NormalizationMethod

from .ensemble_engine import TiledEnsembleEngine
from .ensemble_tiling import EnsembleTiler, TileCollater


def get_ensemble_datamodule(args: dict, tiler: EnsembleTiler, tile_index: tuple[int, int]) -> AnomalibDataModule:
    """Get Anomaly Datamodule adjusted for use in ensemble.

    Datamodule collate function gets replaced by TileCollater in order to tile all images before they are passed on.

    Args:
        args: tiled ensemble run configuration.
        tiler (EnsembleTiler): Tiler used to split the images to tiles for use in ensemble.
        tile_index (tuple[int, int]): Index of the tile in the split image.

    Returns:
        AnomalibDataModule: Anomalib Lightning DataModule
    """
    data_config = args["data"]
    datamodule = get_datamodule(data_config)
    # set custom collate function that does the tiling
    datamodule.collate_fn = TileCollater(tiler, tile_index)
    datamodule.setup()

    return datamodule


def get_ensemble_model(args: dict, tiler: EnsembleTiler) -> AnomalyModule:
    """Get model prepared for ensemble training.

    Args:
        args: tiled ensemble run configuration.
        tiler (EnsembleTiler): tiler used to get tile dimensions.

    Returns:
        AnomalyModule: model with input_size setup
    """
    model_config = args["model"]
    model = get_model(model_config)
    # set model input size match tile size
    model.set_input_size((tiler.tile_size_h, tiler.tile_size_w))

    return model


def get_ensemble_tiler(args: dict) -> EnsembleTiler:
    """Get tiler used for image tiling and to obtain tile dimensions.

    Args:
        args: tiled ensemble run configuration.

    Returns:
        EnsembleTiler: tiler object.
    """
    tiler = EnsembleTiler(
        tile_size=args["ensemble"]["tiling"]["tile_size"],
        stride=args["ensemble"]["tiling"]["stride"],
        image_size=args["data"]["init_args"]["image_size"],
    )

    return tiler  # noqa: RET504


def parse_trainer_kwargs(trainer_args: dict) -> Namespace | dict:
    """Parse trainer args and instantiate all needed elements.

    Transforms config into kwargs ready for Trainer, including instantiation of callback etc.

    Args:
        trainer_args (dict): Trainer args dictionary.

    Returns:
        dict: parsed kwargs with instantiated elements.
    """
    if not trainer_args:
        return {}

    # try to get trainer args, if not present return empty
    parser = ArgumentParser()

    parser.add_class_arguments(Trainer, fail_untyped=False, instantiate=False, sub_configs=True)
    config = parser.parse_object(trainer_args)
    objects = parser.instantiate_classes(config)

    return objects  # noqa: RET504


def get_ensemble_engine(
    tile_index: tuple[int, int],
    accelerator: str,
    devices: list[int] | str | int,
    root_dir: Path,
    normalization_stage: str,
    metrics: dict | None = None,
    trainer_args: dict | None = None,
) -> TiledEnsembleEngine:
    """Prepare engine for ensemble training or prediction.

    This method makes sure correct normalization is used, prepares metrics and additional trainer kwargs..

    Args:
        tile_index (tuple[int, int]): Index of tile that this model processes.
        accelerator (str): Accelerator (device) to use.
        devices (list[int] | str | int): device IDs used for training.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        normalization_stage (str): Config dictionary for ensemble post-processing.
        metrics (dict): Dict containing pixel and image metrics names.
        trainer_args (dict): Trainer args dictionary. Empty dict if not present.

    Returns:
        TiledEnsembleEngine: set up engine for ensemble training/prediction.
    """
    # if we want tile level normalization we set it here, otherwise it's done later on joined images
    if normalization_stage == NormalizationStage.TILE:
        normalization = NormalizationMethod.MIN_MAX
    else:
        normalization = NormalizationMethod.NONE

    # parse additional trainer args and callbacks if present in config
    trainer_kwargs = parse_trainer_kwargs(trainer_args)
    # remove keys that we already have
    trainer_kwargs.pop("accelerator", None)
    trainer_kwargs.pop("default_root_dir", None)
    trainer_kwargs.pop("devices", None)

    # create engine for specific tile location and fit the model
    engine = TiledEnsembleEngine(
        tile_index=tile_index,
        normalization=normalization,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=root_dir,
        image_metrics=metrics.get("image", None) if metrics else None,
        pixel_metrics=metrics.get("pixel", None) if metrics else None,
        **trainer_kwargs,
    )

    return engine  # noqa: RET504