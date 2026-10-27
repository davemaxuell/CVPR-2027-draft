task_solve_with_skills = """{few_shots_section}\
## Skill Manual
The following is the current high-level guidance manual. Treat it as a stable instruction manual for this epoch and follow it throughout the task.

### Main Skill Rules
These are the main reusable task-solving rules. Use them to plan search, object handling, transformations, and failure avoidance.

{main_skills}

### Execution Notes
These notes highlight points that are easy to overlook during execution. Pay special attention to them while applying the main skill rules.

{execution_notes}
---
## Your Turn: Take Action!
Use the above guidance as your foundation, and now work on the following task:
{task_description}
"""

task_few_shots_header = """## Successful Examples (Reference Cases)
Below are some examples of similar tasks that were successfully completed.
Please use these as references to guide your thinking and approach to the current task:

"""

def format_task_prompt_with_skills(
    retrieved_skills: list[str],
    task_description: str,
    skills: list[str] = [],
    few_shots: list[str] = [],
) -> str:
    # Build few-shots section
    few_shots_text = ""
    if few_shots:
        few_shots_text = task_few_shots_header + "\n".join(few_shots) + "\n\n"

    # Build skills section
    combined_skills: list[str] = []
    seen_skills: set[str] = set()
    for skill in list(retrieved_skills or []) + list(skills or []):
        skill_text = str(skill).strip()
        skill_key = skill_text.casefold()
        if not skill_text or skill_key in seen_skills:
            continue
        seen_skills.add(skill_key)
        combined_skills.append(skill_text)

    main_skill_lines: list[str] = []
    execution_note_lines: list[str] = []
    for skill in combined_skills:
        if skill.startswith("[Execution Note"):
            execution_note_lines.append(skill)
        else:
            main_skill_lines.append(skill)

    main_skills_text = '\n'.join([f'{i}. {s}' for i, s in enumerate(main_skill_lines, 1)])
    if not main_skills_text:
        main_skills_text = "(No main skill rules available yet.)"

    execution_notes_text = '\n'.join([f'{i}. {s}' for i, s in enumerate(execution_note_lines, 1)])
    if not execution_notes_text:
        execution_notes_text = "(No execution notes available yet.)"

    user_prompt: str = task_solve_with_skills.format(
        task_description=task_description,
        main_skills=main_skills_text,
        execution_notes=execution_notes_text,
        few_shots_section=few_shots_text,
    )

    return user_prompt
