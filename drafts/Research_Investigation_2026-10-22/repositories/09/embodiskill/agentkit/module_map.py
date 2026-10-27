from typing import Type

from .reasoning import ReasoningBase, ReasoningIO
from .skill import MASSkillBase, EmbodiSkill

def module_map(
    reasoning: str, mas_skill: str = None
) -> tuple[Type[ReasoningBase], Type[MASSkillBase]]:
    
    reasoning_map = {
        'io': ReasoningIO,
    }
    mas_skill_map = {
        'empty': MASSkillBase,
        'embodiskill': EmbodiSkill,
    }

    if reasoning not in reasoning_map:
        raise ValueError(f"Invalid reasoning type '{reasoning}'. Allowed values: {list(reasoning_map.keys())}")

    if mas_skill is not None and mas_skill not in mas_skill_map:
        raise ValueError(f"Invalid skill module type '{mas_skill}'. Allowed values: {list(mas_skill_map.keys())}")

    return (
        reasoning_map[reasoning],
        mas_skill_map.get(mas_skill, None)
    )
    
    
