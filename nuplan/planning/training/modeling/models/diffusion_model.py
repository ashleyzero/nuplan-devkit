from typing import List, Dict, Optional, cast

import torch
from torch import nn

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.lanegcn_utils import (
    Actor2ActorAttention,
    Actor2LaneAttention,
    Lane2ActorAttention,
    LaneNet,
    LinearWithGroupNorm,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentTrafficLightData,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from nuplan.planning.training.preprocessing.feature_builders.critical_agent_feature_builder import CriticalAgentFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.fake_trajectory_target_builder import FakeTrajectoryTargetBuilder
from nuplan.planning.training.preprocessing.features.critical_agent import CriticalAgent
from nuplan.planning.training.modeling.models.diffusion_utils import TransformerConcatLinear, VarianceSchedule, SampleSelector, Constraints, compute_d
from nuplan.planning.training.modeling.models.diffusion_dynamics import Dynamics, DynamicsIntegrateOutput

class DiffusionModel(TorchModuleWrapper):
    """
    Vector-based model that uses a series of MLPs to encode ego and agent signals, a lane graph to encode vector-map
    elements and a fusion network to capture lane & agent intra/inter-interactions through attention layers.
    Dynamic map elements such as traffic light status and ego route information are also encoded in the fusion network.

    Implementation of the original LaneGCN paper ("Learning Lane Graph Representations for Motion Forecasting").
    """

    def __init__(
        self,
        map_net_scales: int,
        num_res_blocks: int,
        num_attention_layers: int,
        a2a_dist_threshold: float,
        l2a_dist_threshold: float,
        num_output_features: int,
        feature_dim: int,
        sampling_method: str,
        sampling_stride: int,
        correction: bool,
        num_corrections: int,
        guidance: bool,
        vector_map_feature_radius: int,
        vector_map_connection_scales: Optional[List[int]],
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        variance_schedule: VarianceSchedule,
        dynamics: Dynamics,
        sample_selector: SampleSelector,
        constraints: Constraints,
    ):
        """
        :param map_net_scales: Number of scales to extend the predecessor and successor lane nodes.
        :param num_res_blocks: Number of residual blocks for the GCN (LaneGCN uses 4).
        :param num_attention_layers: Number of times to repeatedly apply the attention layer.
        :param a2a_dist_threshold: [m] distance threshold for aggregating actor-to-actor nodes
        :param l2a_dist_threshold: [m] distance threshold for aggregating map-to-actor nodes
        :param num_output_features: number of target features
        :param feature_dim: hidden layer dimension
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param vector_map_connection_scales: The hops of lane neighbors to extract, default 1 hop
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(
            feature_builders=[
                CriticalAgentFeatureBuilder(
                    past_trajectory_sampling=past_trajectory_sampling,
                    future_trajectory_sampling=future_trajectory_sampling,
                    radius=vector_map_feature_radius,
                    connection_scales=vector_map_connection_scales,
                ),
            ],
            target_builders=[
                FakeTrajectoryTargetBuilder(),
            ],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        # LaneGCN components
        self.feature_dim = feature_dim
        self.connection_scales = (
            list(range(map_net_scales)) if vector_map_connection_scales is None else vector_map_connection_scales
        )
        # +1 on input dim for both agents and ego to include both history and current steps
        self.ego_input_dim = (past_trajectory_sampling.num_poses + 1) * CriticalAgent.agent_poses_dim()
        self.agent_input_dim = (past_trajectory_sampling.num_poses + 1) * CriticalAgent.agent_states_dim()
        self.lane_net = LaneNet(
            lane_input_len=2,
            lane_feature_len=self.feature_dim,
            num_scales=map_net_scales,
            num_residual_blocks=num_res_blocks,
            is_map_feat=False,
        )
        self.ego_feature_extractor = torch.nn.Sequential(
            nn.Linear(self.ego_input_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            LinearWithGroupNorm(self.feature_dim, self.feature_dim, num_groups=1, activation=False),
        )
        self.agent_feature_extractor = torch.nn.Sequential(
            nn.Linear(self.agent_input_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            LinearWithGroupNorm(self.feature_dim, self.feature_dim, num_groups=1, activation=False),
        )
        self.actor2lane_attention = Actor2LaneAttention(
            actor_feature_len=self.feature_dim,
            lane_feature_len=self.feature_dim,
            num_attention_layers=num_attention_layers,
            dist_threshold_m=l2a_dist_threshold,
        )
        self.lane2actor_attention = Lane2ActorAttention(
            lane_feature_len=self.feature_dim,
            actor_feature_len=self.feature_dim,
            num_attention_layers=num_attention_layers,
            dist_threshold_m=l2a_dist_threshold,
        )
        self.actor2actor_attention = Actor2ActorAttention(
            actor_feature_len=self.feature_dim,
            num_attention_layers=num_attention_layers,
            dist_threshold_m=a2a_dist_threshold,
        )
        # self._mlp = nn.Sequential(
        #     nn.Linear(self.feature_dim, self.feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_dim, self.feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_dim, num_output_features),
        # )
        self.sampling_method = sampling_method
        self.sampling_stride = sampling_stride
        self.correction = correction
        self.num_corrections = num_corrections
        self.guidance = guidance
        self.var_sched = variance_schedule
        self.dynamics = dynamics
        self.sample_selector = sample_selector
        self.constraints = constraints
        self.dt = future_trajectory_sampling.interval_length
        self.num_future_frames = future_trajectory_sampling.num_poses
        # self.e_net = TransformerConcatLinear(context_dim=self.feature_dim, pred_state_dim=self.dynamics.control_dim)
        self.x_0_net = TransformerConcatLinear(context_dim=self.feature_dim, pred_state_dim=self.dynamics.control_dim)

    def encode(self, features: CriticalAgent) -> torch.Tensor:
        """
        LaneGCN's encoder
        :param features: critical agent features
        :return: encoded features
        """
        # Recover features
        vector_map_data = cast(VectorMap, features.vector_map)

        # Extract batches
        batch_size = features.batch_size

        # Extract features
        ego_features = []

        # Map and agent features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):

            sample_ego_feature = self.ego_feature_extractor(
                features.get_history_critical_agent_poses_in_sample(sample_idx).reshape(1, -1))
            sample_ego_center = features.get_critical_agent_center_in_sample(sample_idx)

            # Check for empty vector map input
            if not vector_map_data.is_valid:
                # Create a single lane node located at (0, 0)
                num_coords = 1
                coords = torch.zeros(
                    (num_coords, 2, 2),  # <num_lanes, 2, 2>
                    device=sample_ego_feature.device,
                    dtype=sample_ego_feature.dtype,
                    layout=sample_ego_feature.layout,
                )
                connections = {}
                for scale in self.connection_scales:
                    connections[scale] = torch.zeros((num_coords, 2), device=sample_ego_feature.device).long()
                lane_meta_tl = torch.zeros(
                    (num_coords, LaneSegmentTrafficLightData._encoding_dim), device=sample_ego_feature.device
                )
                lane_meta_route = torch.zeros(
                    (num_coords, LaneOnRouteStatusData._encoding_dim), device=sample_ego_feature.device
                )
                lane_meta = torch.cat((lane_meta_tl, lane_meta_route), dim=1)
            else:
                coords = vector_map_data.coords[sample_idx]
                connections = vector_map_data.multi_scale_connections[sample_idx]
                lane_meta_tl = vector_map_data.traffic_light_data[sample_idx]
                lane_meta_route = vector_map_data.on_route_status[sample_idx]
                lane_meta = torch.cat((lane_meta_tl, lane_meta_route), dim=1)
            lane_features = self.lane_net(coords, connections)
            lane_centers = coords.mean(axis=1)

            # if ego_agent_features.has_agents(sample_idx):
            sample_agents_feature = self.agent_feature_extractor(
                features.get_flatten_surrounding_agents_features_in_sample(sample_idx)
            )
            sample_agents_center = features.get_surrounding_agents_centers_in_sample(sample_idx)
            # else:
            #     # if no agent in the sample, create a single agent with a stationary trajectory at 0s
            #     flattened_agents = torch.zeros(
            #         (1, self.agent_input_dim),
            #         device=sample_ego_feature.device,
            #         dtype=sample_ego_feature.dtype,
            #         layout=sample_ego_feature.layout,
            #     )
            #     sample_agents_feature = self.agent_feature_extractor(flattened_agents)
            #     sample_agents_center = torch.zeros_like(sample_ego_center).unsqueeze(dim=0)

            ego_agents_feature = torch.cat([sample_ego_feature, sample_agents_feature], dim=0)
            ego_agents_center = torch.cat([sample_ego_center.unsqueeze(dim=0), sample_agents_center], dim=0)

            lane_features = self.actor2lane_attention(
                ego_agents_feature, ego_agents_center, lane_features, lane_meta, lane_centers
            )
            ego_agents_feature = self.lane2actor_attention(
                lane_features, lane_centers, ego_agents_feature, ego_agents_center
            )
            ego_agents_feature = self.actor2actor_attention(ego_agents_feature, ego_agents_center)
            ego_features.append(ego_agents_feature[0])

        ego_features = torch.cat(ego_features).view(batch_size, -1)
        return ego_features

    def get_loss(self, features: FeaturesType) -> Optional[float]:
        """
        Predict
        :param features: input features containing
                        {
                            "critical_agent": CriticalAgent,
                        }
        :return: loss: predictions from network
                        {
                            "diffusion_prediction": DiffusionPrediction,
                        }
        """
        critical_agent_features = cast(CriticalAgent, features["critical_agent"])
        if not critical_agent_features.is_valid:
            return None

        context = self.encode(critical_agent_features)

        x_0 = self.dynamics.pack_controls(
            agent=critical_agent_features.get_future_critical_agent(), dt=self.dt).to(device=context.device)
        t = self.var_sched.uniform_sample_t(critical_agent_features.batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].to(device=context.device)

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).to(device=context.device)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).to(device=context.device)   # (B, 1, 1)
        e_random = torch.randn_like(x_0).to(device=context.device)  # (B, N, d)
        # e_theta = self.e_net(c0 * x_0 + c1 * e_random, beta=beta, context=context)
        # loss = torch.nn.functional.mse_loss(
        #     e_theta.view(-1, self.dynamics.control_dim), e_random.view(-1, self.dynamics.control_dim), reduction='mean')
        x_0_theta = self.x_0_net(c0 * x_0 + c1 * e_random, beta=beta, context=context)

        if not self.correction:
            return torch.nn.functional.mse_loss(
                x_0_theta.view(-1, self.dynamics.control_dim), x_0.view(-1, self.dynamics.control_dim), reduction='mean') 

        route_coords = critical_agent_features.get_route_coords()
        present_agent=critical_agent_features.get_present_critical_agent()
        batch_size = critical_agent_features.batch_size
        num_frames = critical_agent_features.num_future_frames

        controls_corrected = x_0_theta
        for i in range(self.num_corrections):
            controls = controls_corrected.detach()
            controls.requires_grad_()
            output = self.dynamics.integrate(
                controls=controls,
                present_agent=present_agent,
                dt=self.dt,
                need_kinematic_values=True,
            )
            g = self.constraints.compute_g(output.poses, output.accelerations, route_coords).sum()
            delta_controls = (torch.autograd.grad(g, controls)[0]).detach()

            if i == self.num_corrections - 1:
                output = self.dynamics.integrate(
                    controls=controls_corrected,
                    present_agent=present_agent,
                    dt=self.dt,
                    need_partial_derivatives=True
                )
                delta_states = torch.bmm(
                    output.partial_derivatives.reshape(-1, self.dynamics.state_dim, self.dynamics.control_dim),
                    delta_controls.reshape(-1, self.dynamics.control_dim, 1)).reshape(batch_size, num_frames, -1)
                poses_corrected = output.poses - delta_states[..., :3] * 1e-3

            controls_corrected = controls_corrected - delta_controls * 1e-3

        poses_target = critical_agent_features.get_future_critical_agent_poses()
        d = compute_d(poses_corrected, poses_target).mean()
        output = self.dynamics.integrate(controls=controls_corrected, present_agent=present_agent, dt=self.dt, need_kinematic_values=True)
        g = self.constraints.compute_g(poses_corrected, output.accelerations, route_coords).mean()
        h = torch.norm(poses_corrected - output.poses, dim=-1).mean()
        loss = d + g*1.0 + h*1.0
        return loss
    
    def sample(self, context: torch.Tensor, critical_agent_features: CriticalAgent) -> torch.Tensor:
        """
        ddpm or ddim sampling
        """
        assert self.sampling_method != "ddim", f"ddim on x0 prediction has not yet been implemented"
        batch_size = context.shape[0]
        x_T = torch.zeros([batch_size, self.num_future_frames, self.dynamics.control_dim]).to(context.device)
        if self.correction or self.guidance:
            route_coords = critical_agent_features.get_route_coords()

        x_t = x_T
        if self.sampling_method == "ddpm":
            for t in range(self.var_sched.num_steps, 0, -1):
                # z = torch.randn_like(x_t).to(context.device) if t > 1 else torch.zeros_like(x_t).to(context.device)
                # alpha = self.var_sched.alphas[t]
                # alpha_bar = self.var_sched.alpha_bars[t]
                # sigma = self.var_sched.get_sigmas(t)
                # c0 = 1.0 / torch.sqrt(alpha).to(context.device)
                # c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar).to(context.device)
                # beta = self.var_sched.betas[[t]*batch_size].to(context.device)
                # e_theta = self.e_net(x_t, beta=beta, context=context)
                # x_t = (c0 * (x_t - c1 * e_theta) + sigma * z).detach()

                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[t-1]
                beta = self.var_sched.betas[[t]*batch_size].to(context.device)
                c0 = torch.sqrt(alpha_bar_next) * (1 - alpha) / (1 - alpha_bar).to(context.device)
                c1 = torch.sqrt(alpha) * (1 - alpha_bar_next) / (1 - alpha_bar).to(context.device)
                x_0_theta = self.x_0_net(x_t, beta=beta, context=context)

                # Correct at every denoising step takes too much time, so we only correct at the final step.
                if self.correction and t == 1:
                    controls_corrected = x_0_theta
                    for i in range(self.num_corrections):
                        with torch.set_grad_enabled(True):
                            controls = controls_corrected.detach()
                            controls.requires_grad_()
                            output = self.dynamics.integrate(
                                controls=controls,
                                present_agent=critical_agent_features.get_present_critical_agent(),
                                dt=self.dt,
                                need_kinematic_values=True,
                            )
                            g = self.constraints.compute_g(output.poses, output.accelerations, route_coords).sum()
                            delta_controls = (torch.autograd.grad(g, controls)[0]).detach()
                            controls_corrected = controls_corrected - delta_controls * 1e-3

                    # x_t = (c0 * controls_corrected + c1 * x_t).detach()
                    x_0_theta = controls_corrected
                
                x_t = (c0 * x_0_theta + c1 * x_t).detach()

                if self.guidance:
                    with torch.set_grad_enabled(True):
                        x_0_theta.requires_grad_()
                        output = self.dynamics.integrate(
                            controls=x_0_theta,
                            present_agent=critical_agent_features.get_present_critical_agent(),
                            dt=self.dt,
                            need_kinematic_values=True,
                        )
                        J = -self.constraints.compute_g(output.poses, output.accelerations, route_coords).sum()
                        grad = (torch.autograd.grad(J, x_0_theta)[0]).detach()
                        x_t += grad * beta

        elif self.sampling_method == "ddim":
            # TODO: implement ddim on x0 prediction
            for t in range(self.var_sched.num_steps, 0, -self.sampling_stride):
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[t-self.sampling_stride]
                beta = self.var_sched.betas[[t]*batch_size].to(context.device)
                e_theta = self.e_net(x_t, beta=beta, context=context)

                x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x_t = (alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta).detach()
        
        return x_t

    def get_metrics(self, features: FeaturesType) -> Optional[Dict[str, float]]:
        """
        Predict trajectory and compute metrics(ADE & FDE)
        :param features: input features containing
                        {
                            "critical_agent": CriticalAgent,
                        }
        :return: metrics: metrics containing
                        {
                            "avg_displacement_error": float,
                            "final_displacement_error": float,
                            "g_route": float,
                            "g_kinematic": float,
                        }
        """
        critical_agent_features = cast(CriticalAgent, features["critical_agent"])
        if not critical_agent_features.is_valid:
            return None

        context = self.encode(critical_agent_features)
        controls = self.sample(context, critical_agent_features)
        output = self.dynamics.integrate(
            controls=controls,
            present_agent=critical_agent_features.get_present_critical_agent(),
            dt=self.dt,
            need_kinematic_values=True
        )

        target_poses = critical_agent_features.get_future_critical_agent_poses()
        avg_displacement_error = torch.norm(output.poses[..., :2] - target_poses[..., :2], dim=-1).mean()
        final_displacement_error = torch.norm(output.poses[:, -1, :2] - target_poses[:, -1, :2], dim=-1).mean()

        route_coords = critical_agent_features.get_route_coords()
        g_route = self.constraints.compute_g_route(output.poses, route_coords).mean()
        g_kinematic = self.constraints.compute_g_kinematic(output.accelerations).mean()
        return {
            "avg_displacement_error": avg_displacement_error,
            "final_displacement_error": final_displacement_error,
            "g_route": g_route,
            "g_kinematic": g_kinematic,
        }
    
    def forward(self, features: FeaturesType) -> TargetsType:
        """
        :param features: input features containing
                        {
                            "critical_agent": CriticalAgent,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                            "velocity": Trajectory,
                        }
        """
        critical_agent_features = cast(CriticalAgent, features["critical_agent"])
        assert critical_agent_features.is_valid, f"feature is not valid"
        context = self.encode(critical_agent_features)
        samples = []
        for i in range(self.sample_selector.num_samples):
            controls = self.sample(context, critical_agent_features)
            samples.append(self.dynamics.integrate(
                controls=controls,
                present_agent=critical_agent_features.get_present_critical_agent(),
                dt=self.dt,
                need_kinematic_values=True))
        route_coords = critical_agent_features.get_route_coords()
        best_indices = self.sample_selector.select_best(samples, route_coords)
        best_poses = []
        best_velocities = []
        for i in range(critical_agent_features.batch_size):
            best_sample = samples[best_indices[i]]
            best_poses.append(best_sample.poses[i])
            best_velocities.append(best_sample.velocities[i])
        return {"trajectory": Trajectory(data=torch.stack(best_poses)),
                "velocity": Trajectory(data=torch.stack(best_velocities))}