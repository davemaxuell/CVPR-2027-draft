AVAILABLE_AGENTS = {
    "baseline_agent": "BaselineAgent",
    "baseline_cot_agent": "BaselineCoTAgent",
    "baseline_reflection_agent": "BaselineReflectionAgent",
    "baseline_cot_reflection_agent": "BaselineCoTReflectionAgent",
    "mapgpt_agent": "MapGPTAgent",
    "mapgpt_cot_agent": "MapGPTCoTAgent",
    "mapgpt_reflection_agent": "MapGPTReflectionAgent",
    "mapgpt_cot_reflection_agent": "MapGPTCoTReflectionAgent"
}

def get_agents(agent_name, logger):
    if agent_name not in AVAILABLE_AGENTS:
        raise ValueError(f"Model {agent_name} not found in available models.")
    
    agent_class = AVAILABLE_AGENTS[agent_name]
    if "." not in agent_class:
        agent_class = f"agents.{agent_name}.{agent_class}"

    try:
        model_module, agent_class = agent_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[agent_class])
        logger.info(f"Agent {agent_class} loaded")
        return getattr(module, agent_class)
    except Exception as e:
        logger.error(f"Failed to import {agent_class} from {agent_name}: {e}")
        raise