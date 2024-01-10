from abc import abstractmethod
from typing import Tuple, Optional
from dataclasses import dataclass

import torch

from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex

@dataclass
class DynamicsIntegrateOutput:
    """
    Stores the results of dynamic integration, as well as the partial derivatives of
    the integration function with respect to the controls.
    """

    poses: torch.tensor  # future poses [batch_size, num_frames, 3]
    velocities: Optional[torch.tensor] = None  # future velocities [batch_size, num_frames, 3]
    accelerations: Optional[torch.tensor] = None  # future accelerations [batch_size, num_frames, 2]
    jerks: Optional[torch.tensor] = None  # future jerks [batch_size, num_frames, 2]
    partial_derivatives: Optional[torch.tensor] = None  # partial derivatives with respect to the controls, [batch_size, num_frames, control_dim]

def gradient(x: torch.tensor) -> torch.tensor:
    """
    Solve the gradient of x in the last dimension. 
     Same as np.gradient, computed using second order accurate central differences in the interior points
     and first order accurate one-sides (forward or backwards) differences at the boundaries.
    :param x: input tensor
    :return: gradient of x
    """
    assert x.shape[-1] >= 2, f"Invalid data tensor! shape of the last dimension: {x.shape[-1]} , expected >= 2"

    return torch.cat((
        (x[..., 1] - x[..., 0]).unsqueeze(-1),
        (x[..., 2:] - x[..., :-2])/2.0,
        (x[..., -1] - x[..., -2]).unsqueeze(-1)), dim=-1)

class Dynamics(object):
    """
    Abstract base class for dynamics model.
    """

    @abstractmethod
    def pack_controls(self, agent: torch.tensor, dt: float) -> torch.tensor:
        """
        pack controls in agent features
        :param agent: [batch_size, num_frames, feature_dim]
        :param dt: time interval
        :return: controls [batch_size, num_frames, control_dim]
        """
        pass

    @abstractmethod
    def integrate(
        self,
        controls: torch.tensor,
        present_agent: torch.tensor,
        dt: float,
        need_kinematic_values: bool = False,
        need_partial_derivatives: bool = False
    ) -> DynamicsIntegrateOutput:
        """
        Solve the future poses based on controls and present agent state.
        :param controls: [batch_size, num_frames, control_dim]
        :param agent: [batch_size, feature_dim]
        :param dt: time interval
        :param need_kinematic_values:
         differentiate pose to obtain velocity, acceleration and jerk if true
        :param need_partial_derivatives: 
         compute partial derivative of the integration function with respect to the controls if true
        :return: future poses, [optional] velocities, accelerations, jerks and partial derivatives
        """
        pass

    @property
    def control_dim(self) -> int:
        pass
    
    @property
    def state_dim(self) -> int:
        pass
    
class SingleIntegrator(Dynamics):

    def pack_controls(self, agent: torch.tensor, dt: float) -> torch.tensor:
        """
        Inherited. See interface.
        """
        assert agent.dim() == 3, f"Got agent ndim: {agent.dim()} , expected 3 [batch_size, num_frames, 8]"

        controls = torch.zeros(agent.shape[0], agent.shape[1], self.control_dim)
        controls[..., SingleIntegratorControlIndex.vx()] = agent[..., AgentFeatureIndex.vx()]
        controls[..., SingleIntegratorControlIndex.vy()] = agent[..., AgentFeatureIndex.vy()]
        return controls

    def integrate(
        self,
        controls: torch.tensor,
        present_agent: torch.tensor,
        dt: float,
        need_kinematic_values: bool = False,
        need_partial_derivatives: bool = False
    ) -> DynamicsIntegrateOutput:
        """
        Inherited. See interface.
        """
        assert controls.shape[-1] == self.control_dim, f"Got control dim: {controls.shape[-1]} , expected {self.control_dim}"
        assert present_agent.dim() == 2, f"Got present agent ndim: {present_agent.dim()} , expected 2 [batch_size, 8]"

        initial_state = torch.zeros(present_agent.shape[0], self.state_dim).to(controls.device) # [batch_size, state_dim]
        initial_state[:, SingleIntegratorStateIndex.x()] = present_agent[:, AgentFeatureIndex.x()]
        initial_state[:, SingleIntegratorStateIndex.y()] = present_agent[:, AgentFeatureIndex.y()]

        positions = torch.cumsum(controls, dim=-2)*dt + initial_state.unsqueeze(1)
        headings = present_agent[:, AgentFeatureIndex.heading()].unsqueeze(1).unsqueeze(2).expand(-1, controls.shape[1], -1)
        output = DynamicsIntegrateOutput(poses=torch.cat((positions, headings), dim=-1))

        if need_kinematic_values:
            accelerations = gradient(torch.norm(controls, dim=-1))/dt
            jerks = gradient(accelerations)/dt
            output.velocities = torch.cat((controls, torch.zeros_like(headings).to(controls.device)), dim=-1)
            output.accelerations = torch.cat((accelerations.unsqueeze(-1), torch.zeros_like(headings).to(controls.device)), dim=-1)
            output.jerks = torch.cat((jerks.unsqueeze(-1), torch.zeros_like(headings).to(controls.device)), dim=-1)
        
        assert need_partial_derivatives == False, f"Partial derivative calculation has not yet been implemented"

        return output

    @property
    def control_dim(self):
        return SingleIntegratorControlIndex.dim()
    
    @property
    def state_dim(self):
        return SingleIntegratorStateIndex.dim()

class Unicycle(Dynamics):

    def pack_controls(self, agent: torch.tensor, dt: float) -> torch.tensor:
        """
        Inherited. See interface.
        """
        assert agent.dim() == 3, f"Got agent ndim: {agent.dim()} , expected 3 [batch_size, num_frames, 8]"

        controls = torch.zeros(agent.shape[0], agent.shape[1], self.control_dim)
        controls[..., UnicycleControlIndex.yaw_rate()] = agent[..., AgentFeatureIndex.yaw_rate()]
        velocities = torch.norm(agent[..., [AgentFeatureIndex.vx(), AgentFeatureIndex.vy()]], dim=-1)
        controls[..., UnicycleControlIndex.a()] = gradient(velocities)/dt
        
        return controls

    def integrate(
        self,
        controls: torch.tensor,
        present_agent: torch.tensor,
        dt: float,
        need_kinematic_values: bool = False,
        need_partial_derivatives: bool = False
    ) -> DynamicsIntegrateOutput:
        """
        Inherited. See interface.
        """
        assert controls.shape[-1] == self.control_dim, f"Got control dim: {controls.shape[-1]} , expected {self.control_dim}"
        assert present_agent.dim() == 2, f"Got present agent ndim: {present_agent.dim()} , expected 2 [batch_size, 8]"

        initial_state = torch.zeros(present_agent.shape[0], self.state_dim).to(controls.device) # [batch_size, state_dim]
        initial_state[:, UnicycleStateIndex.x()] = present_agent[:, AgentFeatureIndex.x()]
        initial_state[:, UnicycleStateIndex.y()] = present_agent[:, AgentFeatureIndex.y()]
        initial_state[:, UnicycleStateIndex.heading()] = present_agent[:, AgentFeatureIndex.heading()]
        initial_state[:, UnicycleStateIndex.v()] = torch.norm(present_agent[:, [AgentFeatureIndex.vx(), AgentFeatureIndex.vy()]], dim=-1)

        states = []
        partial_derivatives = []
        step_state = initial_state
        for i in range(controls.shape[1]):
            step_state, step_partial_derivative = self._step(controls[:, i], step_state, dt, need_partial_derivatives)
            states.append(step_state)
            partial_derivatives.append(step_partial_derivative)
        states = torch.stack(states, dim=1)
        output = DynamicsIntegrateOutput(poses=states[..., :UnicycleStateIndex.heading() + 1])

        if need_kinematic_values:
            vx = states[..., UnicycleStateIndex.v()]*torch.cos(states[..., UnicycleStateIndex.heading()])
            vy = states[..., UnicycleStateIndex.v()]*torch.sin(states[..., UnicycleStateIndex.heading()])
            yaw_rate = states[..., UnicycleControlIndex.yaw_rate()]
            accelerations = gradient(states[..., UnicycleStateIndex.v()])/dt
            jerks = gradient(accelerations)/dt
            angular_accelerations = gradient(yaw_rate)/dt
            angular_jerks = gradient(angular_accelerations)/dt
            output.velocities = torch.stack((vx, vy, yaw_rate), dim=-1)
            output.accelerations = torch.stack((accelerations, angular_accelerations), dim=-1)
            output.jerks = torch.stack((jerks, angular_jerks), dim=-1)
        
        if need_partial_derivatives:
            output.partial_derivatives = torch.stack(partial_derivatives, dim=1)

        return output

    def _step(
            self, control: torch.tensor, state: torch.tensor, dt: float, need_partial_derivative: bool = False
        ) -> Tuple[torch.tensor, Optional[torch.tensor]]:
        """
        https://arxiv.org/pdf/2001.03093.pdf Equation (9)-(11)
        """
        yaw_rate = control[:, UnicycleControlIndex.yaw_rate()]
        a = control[:, UnicycleControlIndex.a()]
        x = state[:, UnicycleStateIndex.x()]
        y = state[:, UnicycleStateIndex.y()]
        heading = state[:, UnicycleStateIndex.heading()]
        v = state[:, UnicycleStateIndex.v()]

        mask_zero = torch.abs(yaw_rate) <= 1e-2
        yaw_rate_nonzero = ~mask_zero*yaw_rate + mask_zero*1.0
        next_heading = heading + yaw_rate_nonzero*dt
        Ds = (torch.sin(next_heading) - torch.sin(heading))/yaw_rate_nonzero # d(sin(heading))/d(yaw_rate)
        Dc = (torch.cos(next_heading) - torch.cos(heading))/yaw_rate_nonzero # d(cos(heading))/d(yaw_rate)

        next_state_zero = torch.stack([x + v*torch.cos(heading)*dt + 0.5*a*torch.cos(heading)*dt**2,
                                       y + v*torch.sin(heading)*dt + 0.5*a*torch.sin(heading)*dt**2,
                                       heading,
                                       v + a*dt], dim=0)
        next_state_nonzero = torch.stack([x + v*Ds + a/yaw_rate_nonzero*Dc + a*torch.sin(next_heading)*dt/yaw_rate_nonzero,
                                          y - v*Dc + a/yaw_rate_nonzero*Ds - a*torch.cos(next_heading)*dt/yaw_rate_nonzero,
                                          next_heading,
                                          v + a*dt], dim=0)
        if not need_partial_derivative:
            return torch.where(mask_zero, next_state_zero, next_state_nonzero).permute(1, 0), None
        
        partial_derivative_zero = torch.stack([torch.zeros_like(x).to(x.device), 0.5*torch.cos(heading)*dt**2,
                                               torch.zeros_like(y).to(y.device), 0.5*torch.sin(heading)*dt**2,
                                               torch.zeros_like(heading).to(heading.device), torch.zeros_like(heading).to(heading.device),
                                               torch.zeros_like(v).to(v.device), torch.ones_like(v).to(v.device)*dt], dim=0)
        
        partial_next_x_partial_yaw_rate = (
            2*a*(torch.cos(heading) - torch.cos(next_heading)) + dt*yaw_rate_nonzero**2*(a*dt + v)*torch.cos(next_heading) + \
            yaw_rate_nonzero*(-2*a*dt*torch.sin(next_heading) + v*(torch.sin(heading) - torch.sin(next_heading))))/yaw_rate_nonzero**3
        partial_next_y_partial_yaw_rate = (
            2*a*(torch.sin(heading) - torch.sin(next_heading)) + dt*yaw_rate_nonzero**2*(a*dt + v)*torch.sin(next_heading) + \
            yaw_rate_nonzero*(2*a*dt*torch.cos(next_heading) - v*(torch.cos(heading) - torch.cos(next_heading))))/yaw_rate_nonzero**3
        partial_derivative_nonzero = torch.stack([partial_next_x_partial_yaw_rate, Dc/yaw_rate_nonzero + torch.sin(next_heading)*dt/yaw_rate_nonzero,
                                                  partial_next_y_partial_yaw_rate, Ds/yaw_rate_nonzero - torch.cos(next_heading)*dt/yaw_rate_nonzero,
                                                  torch.ones_like(heading).to(heading.device)*dt, torch.zeros_like(heading).to(heading.device),
                                                  torch.zeros_like(v).to(v.device), torch.ones_like(v).to(v.device)*dt], dim=0)
        
        return (torch.where(mask_zero, next_state_zero, next_state_nonzero).permute(1, 0),
                torch.where(mask_zero, partial_derivative_zero, partial_derivative_nonzero).permute(1, 0))

    @property
    def control_dim(self):
        return UnicycleControlIndex.dim()
    
    @property
    def state_dim(self):
        return UnicycleStateIndex.dim()

class SingleIntegratorControlIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
        in the controls of single integrator dynamics.

    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the agent.
        :return: index
        """
        return 0

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def dim() -> int:
        """
        The number of controls present in the SingleIntegratorControlIndex.
        :return: number of controls.
        """
        return 2
    
class SingleIntegratorStateIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
        in the states of single integrator dynamics.

    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x coordinate of the agent.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y coordinate of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def dim() -> int:
        """
        The number of states present in the SingleIntegratorStateIndex.
        :return: number of states.
        """
        return 2
    
class UnicycleControlIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
        in the controls of unicycle dynamics.

    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def yaw_rate() -> int:
        """
        The dimension corresponding to the yaw rate of the agent.
        :return: index
        """
        return 0

    @staticmethod
    def a() -> int:
        """
        The dimension corresponding to the acceleration of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def dim() -> int:
        """
        The number of controls present in the SingleIntegratorControlIndex.
        :return: number of controls.
        """
        return 2
    
class UnicycleStateIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
        in the states of unicycle dynamics.

    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x coordinate of the agent.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y coordinate of the agent.
        :return: index
        """
        return 1
    
    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the agent.
        :return: index
        """
        return 2

    @staticmethod
    def v() -> int:
        """
        The dimension corresponding to the velocity of the agent.
        :return: index
        """
        return 3

    @staticmethod
    def dim() -> int:
        """
        The number of states present in the SingleIntegratorStateIndex.
        :return: number of states.
        """
        return 4