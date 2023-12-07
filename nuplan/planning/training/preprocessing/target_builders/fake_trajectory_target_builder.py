from __future__ import annotations

from typing import Type

import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


class FakeTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Fake builders constructed the name."""

    def __init__(self) -> None:
        """
        Initializes the class.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        """
        pass

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """
        :return type of feature which will be generated
        """
        return Trajectory

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "trajectory"

    def get_targets(self, scenario: AbstractScenario) -> AbstractModelFeature:
        """
        Constructs model output targets from database scenario.

        :param scenario: generic scenario
        :return: constructed targets
        """
        return Trajectory(data=np.zeros((1, Trajectory.state_size())))