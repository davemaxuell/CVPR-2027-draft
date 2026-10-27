from agentkit.orchestrator import Orchestrator
from .team import TeamSolver

DEFAULT_WORKFLOW = 'team'


def get_workflow() -> Orchestrator:
    return TeamSolver()
