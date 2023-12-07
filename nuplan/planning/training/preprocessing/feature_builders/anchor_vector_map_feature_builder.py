from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.torch_geometry import coordinates_to_local_frame
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentConnections,
    LaneSegmentCoords,
    LaneSegmentGroupings,
    LaneSegmentTrafficLightData,
    get_neighbor_vector_map,
    get_on_route_status,
    get_traffic_light_encoding,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap

from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder

class AnchorVectorMapFeatureBuilder(VectorMapFeatureBuilder):
    """
    Feature builder for constructing map features in a vector-representation.
    """

    def __init__(self, anchor_state: StateSE2, radius: float, connection_scales: Optional[List[int]] = None) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param anchor_state: anchor state
        :param radius:  The query radius scope relative to the anchor state.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        super().__init__(radius, connection_scales)
        self._anchor_state = anchor_state

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "anchor_vector_map"

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            anchor_coords = Point2D(self._anchor_state.x, self._anchor_state.y)
            (
                lane_seg_coords,
                lane_seg_conns,
                lane_seg_groupings,
                lane_seg_lane_ids,
                lane_seg_roadblock_ids,
            ) = get_neighbor_vector_map(scenario.map_api, anchor_coords, self._radius)

            # compute route following status
            on_route_status = get_on_route_status(scenario.get_route_roadblock_ids(), lane_seg_roadblock_ids)

            # get traffic light status
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                lane_seg_coords,
                lane_seg_conns,
                lane_seg_groupings,
                on_route_status,
                traffic_light_data,
                self._anchor_state,
            )

            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(
                tensors, list_tensors, list_list_tensors
            )

            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)
        
    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            anchor_coords = Point2D(self._anchor_state.x, self._anchor_state.y)
            (
                lane_seg_coords,
                lane_seg_conns,
                lane_seg_groupings,
                lane_seg_lane_ids,
                lane_seg_roadblock_ids,
            ) = get_neighbor_vector_map(initialization.map_api, anchor_coords, self._radius)

            # compute route following status
            on_route_status = get_on_route_status(initialization.route_roadblock_ids, lane_seg_roadblock_ids)

            # get traffic light status
            if current_input.traffic_light_data is None:
                raise ValueError("Cannot build VectorMap feature. PlannerInput.traffic_light_data is None")

            traffic_light_data = current_input.traffic_light_data
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                lane_seg_coords,
                lane_seg_conns,
                lane_seg_groupings,
                on_route_status,
                traffic_light_data,
                self._anchor_state,
            )

            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(
                tensors, list_tensors, list_list_tensors
            )

            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)
        
    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        return {"neighbor_vector_map": {"radius": str(self._radius)}, "anchor_state": empty}
