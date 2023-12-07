from typing import Dict, List, Optional, Tuple, Type, cast

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    compute_yaw_rate_from_state_tensors,
    convert_absolute_quantities_to_relative_to_anchor_state,
    convert_agent_state_to_SE2,
    filter_agents_tensor_appear_in_all_frames,
    select_critical_agent,
    get_critical_agent,
    add_ego_box_states,
    pack_agents_tensor_and_separate_cirtical_agent,
    pad_agent_states,
    sampled_ego_box_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list_with_track_token_dict,
    AgentInternalIndex,
)

from nuplan.planning.training.preprocessing.feature_builders.anchor_vector_map_feature_builder import AnchorVectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.critical_agent import CriticalAgent
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap

class CriticalAgentFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(
        self, past_trajectory_sampling: TrajectorySampling, future_trajectory_sampling: TrajectorySampling,
        radius: float, connection_scales: Optional[List[int]] = None, object_type: TrackedObjectType = TrackedObjectType.VEHICLE
    ) -> None:
        """
        Initializes CriticalAgentFeatureBuilder.
        :param past_trajectory_sampling: Parameters of the sampled past trajectory
        :param future_trajectory_sampling: Parameters of the sampled future trajectory
        :param object_type: Type of agents (TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN) set to TrackedObjectType.VEHICLE by default
        """
        super().__init__()
        if object_type not in [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]:
            raise ValueError(
                f"The model's been tested just for vehicles and pedestrians types, but the provided object_type is {object_type}."
            )
        self.num_past_poses = past_trajectory_sampling.num_poses
        self.past_time_horizon = past_trajectory_sampling.time_horizon
        self.num_future_poses = future_trajectory_sampling.num_poses
        self.future_time_horizon = future_trajectory_sampling.time_horizon
        self.object_type = object_type
        # self._agents_states_dim = Agents.agents_states_dim()

        self._radius = radius
        self._connection_scales = connection_scales

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "critical_agent"

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return CriticalAgent  # type: ignore

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> CriticalAgent:
        """Inherited, see superclass."""
        with torch.no_grad():
            # Retrieve past, present and future ego states and agent boxes
            present_ego_state = scenario.initial_ego_state

            past_ego_states = scenario.get_ego_past_trajectory(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
            future_ego_states = scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
            )
            sampled_ego_states = list(past_ego_states) + [present_ego_state] + list(future_ego_states)
            time_stamps = list(
                scenario.get_past_timestamps(
                    iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
                )
            ) + [scenario.start_time] + list(
                scenario.get_future_timestamps(
                    iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
                )
            )
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]
            sampled_observations = past_tracked_objects + [present_tracked_objects] + future_tracked_objects

            assert len(sampled_ego_states) == len(sampled_observations), (
                "Expected the trajectory length of ego and agent to be equal. "
                f"Got ego: {len(sampled_ego_states)} and agent: {len(sampled_observations)}"
            )

            assert len(sampled_observations) > 2, (
                "Trajectory of length of " f"{len(sampled_observations)} needs to be at least 3"
            )

            tensors, list_tensors, list_list_tensors, track_token_dict = self._pack_to_feature_tensor_dict(
                sampled_ego_states, time_stamps, sampled_observations
            )

            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)

            if not list_tensors:
                empty_vector_map = VectorMap(
                    coords=[], lane_groupings=[], multi_scale_connections=[], on_route_status=[], traffic_light_data=[])
                return CriticalAgent(
                    critical_agent=[],
                    surrounding_agents=[],
                    anchor_state=[],
                    vector_map=empty_vector_map,
                    num_history_frames=self.num_past_poses + 1,
                    num_future_frames=self.num_future_poses
                )
            
            anchor_state = convert_agent_state_to_SE2(list_tensors["anchor_state"][0])
            vector_map_feature_builder = AnchorVectorMapFeatureBuilder(anchor_state, self._radius, self._connection_scales)
            vector_map = vector_map_feature_builder.get_features_from_scenario(scenario)

            output: CriticalAgent = self._unpack_feature_from_tensor_dict(
                tensors, list_tensors, list_list_tensors, vector_map)

            return output

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> CriticalAgent:
        """Inherited, see superclass."""
        with torch.no_grad():
            history = current_input.history
            assert isinstance(
                history.observations[0], DetectionsTracks
            ), f"Expected observation of type DetectionTracks, got {type(history.observations[0])}"

            present_ego_state, present_observation = history.current_state

            past_observations = history.observations[:-1]
            past_ego_states = history.ego_states[:-1]

            assert history.sample_interval, "SimulationHistoryBuffer sample interval is None"

            indices = sample_indices_with_time_horizon(
                self.num_past_poses, self.past_time_horizon, history.sample_interval
            )
            try:
                sampled_past_observations = [
                    cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)
                ]
                sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
            except IndexError:
                raise RuntimeError(
                    f"SimulationHistoryBuffer duration: {history.duration} is "
                    f"too short for requested past_time_horizon: {self.past_time_horizon}. "
                    f"Please increase the simulation_buffer_duration in default_simulation.yaml"
                )

            sampled_past_observations = sampled_past_observations + [
                cast(DetectionsTracks, present_observation).tracked_objects
            ]
            sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
            time_stamps = [state.time_point for state in sampled_past_ego_states]

            tensors, list_tensors, list_list_tensors, track_token_dict = self._pack_to_feature_tensor_dict(
                sampled_past_ego_states, time_stamps, sampled_past_observations
            )
            critical_agent_id = track_token_dict[current_input.critical_agent_track_token]
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors, critical_agent_id)

            anchor_state = convert_agent_state_to_SE2(list_tensors["anchor_state"][0])
            vector_map_feature_builder = AnchorVectorMapFeatureBuilder(anchor_state, self._radius, self._connection_scales)
            vector_map = vector_map_feature_builder.get_features_from_simulation(current_input, initialization)

            output: CriticalAgent = self._unpack_feature_from_tensor_dict(
                tensors, list_tensors, list_list_tensors, vector_map)

            return output

    @torch.jit.unused
    def select_critical_agent_from_scenario(self, scenario: AbstractScenario) -> Optional[str]:
        """
        Select the critical agent from the scenario based on the distance between the ego and the agent's trajectory.
        """
        with torch.no_grad():
            # Retrieve past, present and future ego states and agent boxes
            present_ego_state = scenario.initial_ego_state
            past_ego_states = scenario.get_ego_past_trajectory(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
            future_ego_states = scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
            )
            sampled_ego_states = list(past_ego_states) + [present_ego_state] + list(future_ego_states)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]
            sampled_observations = past_tracked_objects + [present_tracked_objects] + future_tracked_objects

            assert len(sampled_ego_states) == len(sampled_observations), (
                "Expected the trajectory length of ego and agent to be equal. "
                f"Got ego: {len(sampled_ego_states)} and agent: {len(sampled_observations)}"
            )

            assert len(sampled_observations) > 2, (
                "Trajectory of length of " f"{len(sampled_observations)} needs to be at least 3"
            )

            ego_box_states_tensor = sampled_ego_box_states_to_tensor(sampled_ego_states)
            tracked_objects_tensor_list, track_token_dict = sampled_tracked_objects_to_tensor_list_with_track_token_dict(
                tracked_objects=sampled_observations, object_type=self.object_type
            )

            agents = filter_agents_tensor_appear_in_all_frames(tracked_objects_tensor_list)
            if not agents:
                return None
            
            # Used for arranging order
            agent_states = pad_agent_states(agents, reverse=False)
            critical_agent_id = int(agent_states[0][
                select_critical_agent(agent_states, ego_box_states_tensor), AgentInternalIndex.track_token()].item())
            critical_agent_track_token = [track_token 
                                          for track_token, agent_id in track_token_dict.items()
                                          if agent_id == critical_agent_id]
            assert len(critical_agent_track_token) == 1, f"Got {len(critical_agent_track_token)} agents with id {critical_agent_id}"
            return critical_agent_track_token[0]
    
    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        ego_states: List[EgoState],
        time_stamps: List[TimePoint],
        tracked_objects: List[TrackedObjects],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]], Dict[str, int]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param ego_states: The past(& future) states of the ego vehicle.
        :param time_stamps: The past(& future) time stamps of the input data.
        :param tracked_objects: The past(& future) tracked objects.
        :return: The packed tensors.
        """
        # reuse functions in agents_preprocessing.py, which is not limited to the past.
        ego_box_states_tensor = sampled_ego_box_states_to_tensor(ego_states)
        time_stamps_tensor = sampled_past_timestamps_to_tensor(time_stamps)
        tracked_objects_tensor_list, track_token_dict = sampled_tracked_objects_to_tensor_list_with_track_token_dict(
            tracked_objects=tracked_objects, object_type=self.object_type
        )

        return (
            {
                "ego_box_states": ego_box_states_tensor,
                "time_stamps": time_stamps_tensor,
            },
            {"tracked_objects": tracked_objects_tensor_list},
            {},
            track_token_dict,
        )

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
        vector_map: VectorMap,
    ) -> CriticalAgent:
        """
        Unpacks the data returned from the scriptable core into an Agents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed Agents object.
        """
        critical_agent = [list_tensor_data["critical_agent"][0].detach().numpy()]
        surrounding_agents = [list_tensor_data["surrounding_agents"][0].detach().numpy()]
        anchor_state = [list_tensor_data["anchor_state"][0][
            [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]
        ].detach().numpy()]

        num_future_frames = 0 if critical_agent[0].shape[0] == self.num_past_poses + 1 else self.num_future_poses
        return CriticalAgent(
            critical_agent=critical_agent,
            surrounding_agents=surrounding_agents,
            anchor_state=anchor_state,
            vector_map=vector_map,
            num_history_frames=self.num_past_poses + 1,
            num_future_frames=num_future_frames
        )

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
        critical_agent_id: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        ego_box_states: torch.Tensor = tensor_data["ego_box_states"]
        time_stamps: torch.Tensor = tensor_data["time_stamps"]
        agents: List[torch.Tensor] = list_tensor_data["tracked_objects"]

        agents = filter_agents_tensor_appear_in_all_frames(agents)
        if not agents:
            return {}, {}, {}

        # Used for arranging order
        agent_states = pad_agent_states(agents, reverse=False)

        if critical_agent_id is None:
            critical_agent_row = select_critical_agent(agent_states, ego_box_states)
        else:
            critical_agent_row = get_critical_agent(agent_states, critical_agent_id)
        anchor_state = agent_states[self.num_past_poses][critical_agent_row, :].squeeze()

        agent_states = add_ego_box_states(agent_states, ego_box_states)
        critical_agent_row += 1

        local_coords_agent_states = convert_absolute_quantities_to_relative_to_anchor_state(agent_states, anchor_state)

        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(agent_states, time_stamps)

        (critical_agent_tensor, surrounding_agents_tensor) = pack_agents_tensor_and_separate_cirtical_agent(
            local_coords_agent_states, yaw_rate_horizon, critical_agent_row)

        output_dict: Dict[str, torch.Tensor] = {}

        output_list_dict: Dict[str, List[torch.Tensor]] = {
            "critical_agent": [critical_agent_tensor],
            "surrounding_agents": [surrounding_agents_tensor],
            "anchor_state": [anchor_state],
        }

        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}

        return output_dict, output_list_dict, output_list_list_dict

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {
            "past_ego_states": {
                "iteration": "0",
                "num_samples": str(self.num_past_poses),
                "time_horizon": str(self.past_time_horizon),
            },
            "past_time_stamps": {
                "iteration": "0",
                "num_samples": str(self.num_past_poses),
                "time_horizon": str(self.past_time_horizon),
            },
            "past_tracked_objects": {
                "iteration": "0",
                "time_horizon": str(self.past_time_horizon),
                "num_samples": str(self.num_past_poses),
            },
        }
