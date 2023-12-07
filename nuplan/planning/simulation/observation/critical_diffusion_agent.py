from abc import abstractmethod
from typing import List, Dict, Optional, Type, cast

from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects, TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import sort_dict

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.training.preprocessing.features.critical_agent import CriticalAgent
from nuplan.common.actor_state.oriented_box import OrientedBox


class CriticalDiffusionAgent(AbstractObservation):
    """
    Simulate critical agent based on a diffusion model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario) -> None:
        """
        Initializes the AbstractEgoCentricMLAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        self._model_loader = ModelLoader(model)
        self._feature_builder = self._model_loader.feature_builders[0]
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time
        # self._num_output_dim = model.future_trajectory_sampling.num_poses

        self._scenario = scenario
        self._anchor_state = None

        # self.step_time = None  # time pass since last simulation iteration
        self._current_iteration = 0
        # self._agents: Optional[Dict[str, TrackedObject]] = None
        self._critical_agent_track_token: Optional[str] = None
        self._critical_agent: Optional[Agent] = None

    def _initialize_critical_agent(self) -> None:
        """
        Initializes the critical agent based on the first step of the scenario
        """
        # unique_agents = {
        #     tracked_object.track_token: tracked_object
        #     for tracked_object in self._scenario.initial_tracked_objects.tracked_objects
        #     if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE
        # }
        # # TODO: consider agents appearing in the future (not just the first frame)
        # self._agents = sort_dict(unique_agents)

        self._critical_agent_track_token = self._feature_builder.select_critical_agent_from_scenario(self._scenario)
        assert self._critical_agent_track_token is not None, f"No critical agent in scenario"
        print("critical agent track token: ", self._critical_agent_track_token)

        for tracked_object in self._scenario.initial_tracked_objects.tracked_objects:
            if tracked_object.track_token == self._critical_agent_track_token:
                self._critical_agent = Agent(
                tracked_object_type=TrackedObjectType.CRITICAL_AGENT,
                oriented_box=tracked_object.box,
                velocity=tracked_object.velocity,
                metadata=tracked_object.metadata,
            )
        assert self._critical_agent is not None, f"Do not get agent with track token in scenario"

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._initialize_critical_agent()

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._initialize_critical_agent()
        self._model_loader.initialize()

    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        # self.step_time = next_iteration.time_point - iteration.time_point
        # self._ego_anchor_state, _ = history.current_state
        self._current_iteration = next_iteration.index
        if self._critical_agent_track_token is None:
            return

        # Construct input features
        # TODO: Rename PlannerInitialization to something that also applies to smart agents
        initialization = PlannerInitialization(
            mission_goal=self._scenario.get_mission_goal(),
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            map_api=self._scenario.map_api,
        )
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(next_iteration.index)
        current_input = PlannerInput(next_iteration, history, traffic_light_data, self._critical_agent_track_token)
        features = self._model_loader.build_features(current_input, initialization)
        predictions = self._model_loader.infer(features)

        trajectory = cast(Trajectory, predictions["trajectory"])
        velocity = cast(Trajectory, predictions["velocity"])
        critical_agent = cast(CriticalAgent, features["critical_agent"])
        anchor_pose, anchor_velocity = critical_agent.get_anchor_state()

        relative_pose = StateSE2(trajectory.xy[0, 0, 0].item(), trajectory.xy[0, 0, 1].item(), trajectory.heading[0, 0].item())
        relative_velocity = StateSE2(velocity.xy[0, 0, 0].item(), velocity.xy[0, 0, 1].item(), trajectory.heading[0, 0].item())
        print("iter ", self._current_iteration)
        absoulte_pose = relative_to_absolute_poses(anchor_pose, [relative_pose])[0]
        absoulte_velocity = relative_to_absolute_poses(anchor_velocity, [relative_velocity])[0]
        print("pose ", absoulte_pose, " vel ", absoulte_velocity)

        self._critical_agent = Agent(
                tracked_object_type=self._critical_agent.tracked_object_type,
                oriented_box=OrientedBox.from_new_pose(self._critical_agent.box, absoulte_pose),
                velocity=StateVector2D(absoulte_velocity.x, absoulte_velocity.y),
                metadata=self._critical_agent.metadata,
            )

        
    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        observations = self._scenario.get_tracked_objects_at_iteration(self._current_iteration)
        if self._critical_agent_track_token is None:
            return observations

        agents: List[TrackedObject] = []
        critical_agent_in_agents = False
        for tracked_object in observations.tracked_objects:
            if tracked_object.track_token == self._critical_agent_track_token:
                # tracked_object = self._critical_agent
                agents.append(self._critical_agent)
                critical_agent_in_agents = True
            else:
                agents.append(tracked_object)
        if not critical_agent_in_agents:
            agents.append(self._critical_agent)

        return DetectionsTracks(TrackedObjects(agents))
