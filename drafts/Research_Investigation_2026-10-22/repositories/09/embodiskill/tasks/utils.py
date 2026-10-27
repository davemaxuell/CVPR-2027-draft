import re


def get_model_type(model_name: str) -> str:
    
    valid_model_types: list[str] = [
        'gpt-4o-mini', 
        'gpt-5',
        'qwen2.5-7b', 
        'qwen2.5-14b-3',
        'qwen2.5-14b-2',
        'qwen2.5-14b',
        'qwen2.5-32b', 
        'qwen2.5-72b',
        'intern', 
        'deepseek-v3',
        'gemini-2.5-flash',
        'gemini-2.5-pro',
    ]

    for model_type in valid_model_types:
        if model_type in model_name.lower():
            return model_type
    
    return 'unknown'


def sanitize_model_name(model_name: str) -> str:
    """Convert a model name into a filesystem-safe directory segment."""
    return re.sub(r'[^A-Za-z0-9._-]+', '-', model_name.strip()).strip('-') or 'unknown'


def get_model_dir_name(agent_model_name: str, skill_model_name: str = None) -> str:
    """Return a directory-safe tag that preserves both model names."""
    agent_model_dir = sanitize_model_name(agent_model_name)
    skill_model_dir = sanitize_model_name(skill_model_name or agent_model_name)
    return f"agent-{agent_model_dir}_skill-{skill_model_dir}"
