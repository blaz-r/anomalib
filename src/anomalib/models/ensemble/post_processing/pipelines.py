import logging
from typing import List

from omegaconf import DictConfig, ListConfig

from anomalib.models.ensemble.ensemble_functions import BasicPredictionJoiner
from anomalib.models.ensemble.ensemble_prediction_data import EnsemblePredictions
from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler
from anomalib.models.ensemble.post_processing import (
    EnsembleMetrics,
    EnsembleVisualization,
    EnsemblePostProcessPipeline,
    SmoothJoins,
    PostProcessStats,
    EnsemblePostProcess,
    MinMaxNormalize,
    Threshold,
)
from anomalib.post_processing import ThresholdMethod


logger = logging.getLogger(__name__)


def get_stats_pipeline(config: DictConfig | ListConfig, tiler: EnsembleTiler) -> EnsemblePostProcessPipeline:
    """
    Construct pipeline used to obtain prediction statistics.

    Args:
        config: Configurable parameters object.
        tiler: Tiler used by some steps of pipeline.

    Returns:
        Constructed pipeline.
    """
    stats_pipeline = EnsemblePostProcessPipeline(BasicPredictionJoiner(tiler))

    steps = []

    if config.ensemble.post_processing.smooth_joins:
        steps.append(SmoothJoins(tiler))
    if config.ensemble.metrics.threshold.method == ThresholdMethod.ADAPTIVE:
        if (
            config.ensemble.metrics.threshold.stage == "final"
            or config.ensemble.post_processing.normalization == "final"
        ):
            # thresholding and normalization need joined image-level statistics
            steps.append(PostProcessStats())

    stats_pipeline.add_steps(steps)

    return stats_pipeline


def get_stats(
    config: DictConfig | ListConfig, tiler: EnsembleTiler, validation_predictions: EnsemblePredictions
) -> dict:
    """
    Get statistics used for postprocessing.

    Args:
        config: Configurable parameters object.
        tiler: Tiler used by some steps of pipeline.
        validation_predictions: Predictions used to calculate stats.

    Returns:
        Dictionary with calculated statistics.
    """
    stats_pipeline = get_stats_pipeline(config, tiler)

    pipe_out = stats_pipeline.execute(validation_predictions)

    return pipe_out.get("stats", None)


def log_postprocess_steps(steps: List[EnsemblePostProcess]) -> None:
    """
    Log steps used in post-processing pipeline.

    Args:
        steps: List of steps in pipeline.

    """
    logger.info("-" * 42)
    logger.info("Steps in post processing pipeline:")
    for step in steps:
        logger.info(step.name)
    logger.info("-" * 42)


def get_postprocessing_pipeline(
    config: DictConfig | ListConfig, tiler: EnsembleTiler, stats: dict
) -> EnsemblePostProcessPipeline:
    """
    Construct pipeline used to post process ensemble predictions.

    Args:
        config: Configurable parameters object.
        tiler: Tiler used by some steps of pipeline.
        stats: Statistics of predictions (min, max, thresholds).

    Returns:
        Constructed pipeline.
    """
    post_pipeline = EnsemblePostProcessPipeline(BasicPredictionJoiner(tiler))

    steps = []
    if config.ensemble.post_processing.smooth_joins:
        steps.append(SmoothJoins(tiler))

    # if normalization is done at the end on image-level
    if config.ensemble.post_processing.normalization == "final":
        steps.append(MinMaxNormalize(stats))
        # with minmax normalization, values are normalized such that the threshold value is centered at 0.5
        stats["image_threshold"] = 0.5
        stats["pixel_threshold"] = 0.5

    # override threshold if it's set manually
    if config.ensemble.metrics.threshold.method == ThresholdMethod.MANUAL:
        stats["image_threshold"] = config.ensemble.metrics.threshold.manual_image
        stats["pixel_threshold"] = config.ensemble.metrics.threshold.manual_pixel

    # if thresholding is done at the end on image-level
    if config.ensemble.metrics.threshold.stage == "final":
        steps.append(Threshold(stats["image_threshold"], stats["pixel_threshold"]))

    if config.ensemble.visualization.show_images or config.ensemble.visualization.save_images:
        steps.append(EnsembleVisualization(config))

    steps.append(EnsembleMetrics(config, stats["image_threshold"], stats["pixel_threshold"]))
    post_pipeline.add_steps(steps)

    log_postprocess_steps(steps)

    return post_pipeline


def post_process(
    config: DictConfig | ListConfig,
    tiler: EnsembleTiler,
    ensemble_predictions: EnsemblePredictions,
    validation_predictions: EnsemblePredictions,
) -> dict:
    """
    Postprocess, visualize and calculate metrics.

    Args:
        config: Configurable parameters object.
        tiler: Tiler used for untiling of predictions.
        ensemble_predictions: Predictions to be joined and processed.
        validation_predictions: Predictions used to calculate stats.

    Returns:
        Dictionary with calculated metrics data.
    """
    logger.info("Computing normalization and threshold statistics.")
    # get statistics, calculated on validation dataset
    stats = get_stats(config, tiler, validation_predictions)

    post_pipeline = get_postprocessing_pipeline(config, tiler, stats)

    logger.info("Executing pipeline.")
    # add all above configured steps to pipeline and execute
    pipe_out = post_pipeline.execute(ensemble_predictions)

    return pipe_out["metrics"]
