"""Base class for thresholding metrics."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Any

from torchmetrics import Metric


class BaseThreshold(Metric, ABC):
    """Base class for thresholding metrics."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def compute(self) -> Any:
        """Compute the threshold

        Returns:
            Value of the optimal threshold.
        """
        raise NotImplementedError("Subclass of BaseAnomalyScoreThreshold must implement the compute method")

    def update(self, *args, **kwargs) -> None:
        """Update the metric state

        Args:
            *args: Any positional arguments.
            **kwargs: Any keyword arguments.
        """
        raise NotImplementedError("Subclass of BaseAnomalyScoreThreshold must implement the update method")