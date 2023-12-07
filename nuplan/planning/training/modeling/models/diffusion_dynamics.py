from abc import abstractmethod
from typing import Tuple

import torch

from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex

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
    def integrate(self, controls: torch.tensor, present_agent: torch.tensor, dt: float) -> torch.tensor:
        """
        Solve the future poses based on controls and present agent state.
        :param controls: [batch_size, num_frames, control_dim]
        :param agent: [batch_size, feature_dim]
        :param dt: time interval
        :return: future poses
        """
        pass

    @abstractmethod
    def integrate_and_differentiate(
        self, controls: torch.tensor, present_agent: torch.tensor, dt: float
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Solve the future poses based on controls and present agent state, and differentiate to obtain velocity, acceleration and jerk
        :param controls: [batch_size, num_frames, control_dim]
        :param agent: [batch_size, feature_dim]
        :param dt: time interval
        :return: future poses [batch_size, num_frames, 3], velocities [batch_size, num_frames, 3], 
         accelerations [batch_size, num_frames, 2], jerks [batch_size, num_frames, 2]
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

    def integrate(self, controls: torch.tensor, present_agent: torch.tensor, dt: float) -> torch.tensor:
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

        return torch.cat((positions, headings), dim=-1)
    
    def integrate_and_differentiate(
            self, controls: torch.tensor, present_agent: torch.tensor, dt: float
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
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
        accelerations = gradient(torch.norm(controls, dim=-1))/dt
        jerks = gradient(accelerations)/dt
        
        return (
            torch.cat((positions, headings), dim=-1), 
            torch.cat((controls, torch.zeros_like(headings).to(controls.device)), dim=-1),
            torch.cat((accelerations.unsqueeze(-1), torch.zeros_like(headings).to(controls.device)), dim=-1),
            torch.cat((jerks.unsqueeze(-1), torch.zeros_like(headings).to(controls.device)), dim=-1)
        )

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

    def integrate(self, controls: torch.tensor, present_agent: torch.tensor, dt: float) -> torch.tensor:
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
        step_state = initial_state
        for i in range(controls.shape[1]):
            step_state = self._step(controls[:, i], step_state, dt)
            states.append(step_state)
        states = torch.stack(states, dim=1)

        return states[..., :UnicycleStateIndex.heading() + 1]
    
    def integrate_and_differentiate(
            self, controls: torch.tensor, present_agent: torch.tensor, dt: float
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
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
        step_state = initial_state
        for i in range(controls.shape[1]):
            step_state = self._step(controls[:, i], step_state, dt)
            states.append(step_state)
        states = torch.stack(states, dim=1)

        vx = states[..., UnicycleStateIndex.v()]*torch.cos(states[..., UnicycleStateIndex.heading()])
        vy = states[..., UnicycleStateIndex.v()]*torch.sin(states[..., UnicycleStateIndex.heading()])
        yaw_rate = states[..., UnicycleControlIndex.yaw_rate()]

        accelerations = gradient(states[..., UnicycleStateIndex.v()])/dt
        jerks = gradient(accelerations)/dt
        angular_accelerations = gradient(yaw_rate)/dt
        angular_jerks = gradient(angular_accelerations)/dt

        return (
            states[..., :UnicycleStateIndex.heading() + 1],
            torch.stack((vx, vy, yaw_rate), dim=-1),
            torch.stack((accelerations, angular_accelerations), dim=-1),
            torch.stack((jerks, angular_jerks), dim=-1)
        )

    def _step(self, control: torch.tensor, state: torch.tensor, dt: float) -> torch.tensor:
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
        
        return torch.where(mask_zero, next_state_zero, next_state_nonzero).permute(1, 0)

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