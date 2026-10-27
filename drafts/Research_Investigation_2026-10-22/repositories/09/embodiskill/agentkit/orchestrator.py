from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any

from .agents import Agent, Env
from .reasoning import ReasoningBase
from .skill import MASSkillBase

@dataclass
class Orchestrator:
    """Unified management of members in the MAS system to achieve overall scheduling of the MAS.
    """
    agents_team: Dict[str, Agent] = field(default_factory=dict)  
    env: Optional[Env] = None  
    meta_skill: Optional[MASSkillBase] = None  
    
    def hire(self, agents: Iterable[Agent]) -> None:
        for agent in agents:
            if agent.name not in self.agents_team:
                self.agents_team[agent.name] = agent  
            else:
                print(f"Agent {agent.name} is already in the team.")  

    def set_env(self, env: Env) -> None:
        self.env = env
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents_team.get(agent_name)
    
    @abstractmethod
    def build_system(self, reasoning: ReasoningBase, mas_skill: MASSkillBase) -> Any:
        pass
    
    @abstractmethod
    def schedule(self, **args) -> Any:
        pass
