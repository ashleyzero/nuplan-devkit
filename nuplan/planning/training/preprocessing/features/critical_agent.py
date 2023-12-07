from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Tuple, cast

import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)

from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.common.actor_state.state_representation import StateSE2

@dataclass
class CriticalAgent(AbstractModelFeature):
    """
    Model input feature representing the past, present and future states of the critical agent and surrounding agents.
    The ego agent is placed first among the surrounding agents.

        critical_agent: List[<np.ndarray: num_frames, 8>].
            The outer list is the batch dimension.
            The num_frames includes both past, present and future frames.
            The last dimension is the critical agent pose (x, y, heading) velocities (vx, vy, yaw rate)
             and size (length, width) at time t.
            Example dimensions: 8 (batch_size) x 21 (4 past + 1 present + 16 future frames) x 8
        surrounding_agents: List[<np.ndarray: num_frames, num_agents, 8>].
            The outer list is the batch dimension.
            The num_frames includes both past, present and future frames.
            The num_agents is filtered to keep only agents which appear in all frames.
            The last dimension is the critical agent pose (x, y, heading) velocities (vx, vy, yaw rate)
             and size (length, width) at time t.
        anchor_state: List[<np.ndarray: num_frames, 3>].
            The outer list is the batch dimension.
            The num_frames includes both past, present and future frames.
            The last dimension is the anchor state (x, y, heading)

    The past/present/future frames dimension is populated in increasing chronological order,
    i.e. (t_-N, ..., t_-1, t_0, t_1, ..., t_M), where N + 1 + M is the number of frames in the feature

    In both cases, the outer List represent number of batches. This is a special feature where each batch entry
    can have different size. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """

    critical_agent: List[FeatureDataType]
    surrounding_agents: List[FeatureDataType]
    anchor_state: List[FeatureDataType]
    vector_map: VectorMap
    num_history_frames: int
    num_future_frames: int

    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        if len(self.critical_agent) != len(self.surrounding_agents):
            raise AssertionError(f"Not consistent length of batches! {len(self.critical_agent)} != {len(self.surrounding_agents)}")

        if self.batch_size != 0 and self.critical_agent[0].ndim != 2:
            raise AssertionError(
                "CriticalAgent feature samples does not conform to feature dimensions! "
                f"Got ndim: {self.critical_agent[0].ndim} , expected 2 [num_frames, 8]"
            )

        if self.batch_size != 0 and self.surrounding_agents[0].ndim != 3:
            raise AssertionError(
                "SurroundingAgents feature samples does not conform to feature dimensions! "
                f"Got ndim: {self.surrounding_agents[0].ndim} , expected 3 [num_frames, num_agents, 8]"
            )

        for i in range(len(self.critical_agent)):
            if int(self.critical_agent[i].shape[0]) != self.num_frames or int(self.surrounding_agents[i].shape[0]) != self.num_frames:
                raise AssertionError("CriticalAgent feature samples have different number of frames!")
            
        assert isinstance(self.vector_map, VectorMap), "vector_map is not an instance of VectorMap"

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return (
            len(self.critical_agent) > 0
            and len(self.surrounding_agents) > 0
            and len(self.critical_agent) == len(self.surrounding_agents)
            and len(self.critical_agent) == len(self.anchor_state)
            and len(self.critical_agent[0]) > 0
            and len(self.surrounding_agents[0]) > 0
            and len(self.critical_agent[0]) == len(self.surrounding_agents[0]) > 0
            and self.critical_agent[0].shape[-1] == self.agent_states_dim()
            and self.surrounding_agents[0].shape[-1] == self.agent_states_dim()
        )

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        return len(self.critical_agent)

    @classmethod
    def collate(cls, batch: List[CriticalAgent]) -> CriticalAgent:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return CriticalAgent(
            critical_agent=[item.critical_agent[0] for item in batch if item.batch_size == 1],
            surrounding_agents=[item.surrounding_agents[0] for item in batch if item.batch_size == 1],
            anchor_state=[item.anchor_state[0] for item in batch if item.batch_size == 1],
            vector_map=VectorMap.collate([item.vector_map for item in batch if item.batch_size == 1]),
            num_history_frames=batch[0].num_history_frames,
            num_future_frames=batch[0].num_future_frames,
        )

    def to_feature_tensor(self) -> CriticalAgent:
        """Implemented. See interface."""
        return CriticalAgent(
            critical_agent=[to_tensor(critical_agent) for critical_agent in self.critical_agent],
            surrounding_agents=[to_tensor(surrounding_agents) for surrounding_agents in self.surrounding_agents],
            anchor_state=[to_tensor(anchor_state) for anchor_state in self.anchor_state],
            vector_map=self.vector_map.to_feature_tensor(),
            num_history_frames=self.num_history_frames,
            num_future_frames=self.num_future_frames,
        )

    def to_device(self, device: torch.device) -> CriticalAgent:
        """Implemented. See interface."""
        return CriticalAgent(
            critical_agent=[to_tensor(critical_agent).to(device=device) for critical_agent in self.critical_agent],
            surrounding_agents=[to_tensor(surrounding_agents).to(device=device) for surrounding_agents in self.surrounding_agents],
            anchor_state=[to_tensor(anchor_state).to(device=device) for anchor_state in self.anchor_state],
            vector_map=self.vector_map.to_device(device),
            num_history_frames=self.num_history_frames,
            num_future_frames=self.num_future_frames,
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> CriticalAgent:
        """Implemented. See interface."""
        return CriticalAgent(
            critical_agent=data["critical_agent"],
            surrounding_agents=data["surrounding_agents"],
            anchor_state=data["anchor_state"],
            vector_map=VectorMap.deserialize(data["vector_map"]),
            num_history_frames=data["num_history_frames"],
            num_future_frames=data["num_future_frames"],
        )

    def unpack(self) -> List[CriticalAgent]:
        """Implemented. See interface."""
        vector_maps = self.vector_map.unpack()
        return [CriticalAgent([critical_agent], [surrounding_agents], [anchor_state],
                              vector_map, self.num_history_frames, self.num_future_frames) 
                for critical_agent, surrounding_agents, anchor_state, vector_map
                in zip(self.critical_agent, self.surrounding_agents, vector_maps)]

    def num_surrounding_agents_in_sample(self, sample_idx: int) -> int:
        """
        Returns the number of surrounding_agents at a given batch
        :param sample_idx: the batch index of interest
        :return: number of surrounding_agents in the given batch
        """
        return self.surrounding_agents[sample_idx].shape[1]  # type: ignore

    @staticmethod
    def agent_poses_dim() -> int:
        """
        :return: agent pose(x, y, heading) dimension
        """
        return 3
    
    @staticmethod
    def agent_states_dim() -> int:
        """
        :return: agent state dimension
        """
        return AgentFeatureIndex.dim()
    
    @property
    def num_frames(self) -> int:
        """
        :return: number of frames.
        """
        return self.num_history_frames + self.num_future_frames

    @property
    def agent_feature_dim(self) -> int:
        """
        :return: agent feature dimension.
        """
        return CriticalAgent.agent_states_dim() * self.num_history_frames

    def get_flatten_surrounding_agents_features_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Flatten surrounding agents' features by stacking the agents' states along the num_frame dimension
        <np.ndarray: num_frames, num_agents, 8>] -> <np.ndarray: num_agents, num_frames x 8>]

        :param sample_idx: the sample index of interest
        :return: <FeatureDataType: num_agents, num_frames x 8>] agent feature
        """
        # if self.num_agents_in_sample(sample_idx) == 0:
        #     if isinstance(self.ego[sample_idx], torch.Tensor):
        #         return torch.empty(
        #             (0, self.num_frames * AgentFeatureIndex.dim()),
        #             dtype=self.ego[sample_idx].dtype,
        #             device=self.ego[sample_idx].device,
        #         )
        #     else:
        #         return np.empty(
        #             (0, self.num_frames * AgentFeatureIndex.dim()),
        #             dtype=self.ego[sample_idx].dtype,
        #         )

        data = self.surrounding_agents[sample_idx][:self.num_history_frames]
        axes = (1, 0) if isinstance(data, torch.Tensor) else (1, 0, 2)
        return data.transpose(*axes).reshape(data.shape[1], -1)

    def get_history_critical_agent_poses_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns the history critical agent's poses in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_history_frames, num_agents, 3>. (x, y, heading) poses of the critical agent's center at the sample index
        """
        return self.critical_agent[sample_idx][:self.num_history_frames, :AgentFeatureIndex.heading() + 1]
    
    def get_present_critical_agent_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present critical agent in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 8>. critical agent at sample index
        """
        return self.critical_agent[sample_idx][self.num_history_frames - 1]
    
    def get_present_surrounding_agents_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present surrounding agents' states in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_agents, 8>. present surrounding agents states at sample index
        """
        return self.surrounding_agents[sample_idx][self.num_history_frames - 1]
    
    def get_critical_agent_center_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return critical agent's center in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 2>. (x, y) positions of the critical agent's center at sample index
        """
        return self.get_present_critical_agent_in_sample(sample_idx)[: AgentFeatureIndex.y() + 1]

    def get_surrounding_agents_centers_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns surrounding agents'centers in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_agents, 2>. (x, y) positions of the surrounding agents' centers at the sample index
        """
        return self.get_present_surrounding_agents_in_sample(sample_idx)[:, : AgentFeatureIndex.y() + 1]

    # def get_surrounding_agents_lengths_in_sample(self, sample_idx: int) -> FeatureDataType:
    #     """
    #     Returns surrounding agents' lengths in the given sample index
    #     :param sample_idx: the batch index of interest
    #     :return: <FeatureDataType: num_agents>. lengths of all the surrounding agents at the sample index
    #     """
    #     return self.get_present_surrounding_agents_in_sample(sample_idx)[:, AgentFeatureIndex.length()]

    # def get_surrounding_agents_widths_in_sample(self, sample_idx: int) -> FeatureDataType:
    #     """
    #     Returns surrounding agents' widths in the given sample index
    #     :param sample_idx: the batch index of interest
    #     :return: <FeatureDataType: num_agents>. width of the surrounding agents at the sample index
    #     """
    #     return self.get_present_surrounding_agents_in_sample(sample_idx)[:, AgentFeatureIndex.width()]
    
    # def get_surrounding_agents_corners_in_sample(self, sample_idx: int) -> FeatureDataType:
    #     """
    #     Returns surrounding agents' corners in the given sample index
    #     :param sample_idx: the batch index of interest
    #     :return: <FeatureDataType: num_agents, 4, 3>. (x, y, 1) positions of the surrounding agents' corners at the sample index
    #     """
    #     widths = self.get_surrounding_agents_lengths_in_sample(sample_idx)
    #     lengths = self.get_surrounding_agents_widths_in_sample(sample_idx)

    #     half_widths = widths / 2.0
    #     half_lengths = lengths / 2.0

    #     feature_cls = np.array if isinstance(widths, np.ndarray) else torch.Tensor

    #     return feature_cls(
    #         [
    #             [
    #                 [half_length, half_width, 1.0],
    #                 [-half_length, half_width, 1.0],
    #                 [-half_length, -half_width, 1.0],
    #                 [half_length, -half_width, 1.0],
    #             ]
    #             for half_width, half_length in zip(half_widths, half_lengths)
    #         ]
    #     )

    def get_present_critical_agent(self) -> FeatureDataType:
        """
        Returns the present critical agent
        :return: <FeatureDataType: num_batches, 8>.
        """
        present_critical_agent = [critical_agent_sample[self.num_history_frames - 1] for critical_agent_sample in self.critical_agent]
        return torch.stack(present_critical_agent)

    def get_future_critical_agent(self) -> FeatureDataType:
        """
        Returns the future critical agent
        :return: <FeatureDataType: num_batches, num_future_frames, 8>.
        """
        future_critical_agent = [critical_agent_sample[self.num_history_frames:] for critical_agent_sample in self.critical_agent]
        return torch.stack(future_critical_agent)

    def get_future_critical_agent_poses(self) -> FeatureDataType:
        """
        Returns the future critical agent poses
        :return: <FeatureDataType: num_batches, num_future_frames, 3>.
        """
        return self.get_future_critical_agent()[:, :, :AgentFeatureIndex.heading() + 1]
    
    def get_anchor_state(self) -> Tuple[StateSE2, StateSE2]:
        assert self.batch_size == 1, f"Expected batch size be 1, got {self.batch_size}"
        anchor_pose = StateSE2(float(self.anchor_state[0][0].item()), float(self.anchor_state[0][1].item()), float(self.anchor_state[0][2].item()))
        anchor_velocity = StateSE2(0.0, 0.0, float(self.anchor_state[0][2].item()))
        return anchor_pose, anchor_velocity
    
    def get_route_coords(self) -> List[FeatureDataType]:
        """
        element in route_coords may be torch.tensor([])
        """
        route_coords = []
        on_route_code = torch.tensor([1.0, 0.0]).to(self.vector_map.on_route_status[0].device)
        for i in range(self.batch_size):
            coords = self.vector_map.coords[i]
            on_route_status = self.vector_map.on_route_status[i]
            route_coords.append(coords[on_route_status == on_route_code])
        return route_coords
