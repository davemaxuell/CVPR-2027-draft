from dataclasses import dataclass


def normalize_manual_profile(task_name: str | None) -> str:
    raw = str(task_name or "").strip().lower()
    if "alfworld" in raw or "alfred" in raw:
        return "alfworld"
    return "alfworld"


def get_manual_section_templates(task_profile: str) -> list[dict[str, list[str]]]:
    titles = get_manual_section_titles(task_profile)
    return [{"title": title, "items": []} for title in titles]


def get_manual_section_titles(task_profile: str) -> list[str]:
    return [
        "Planning",
        "Search",
        "Execution",
        "Failure Avoidance",
    ]


def get_manual_display_name(task_profile: str) -> str:
    return "ALFWorld Skill Manual"

# ---------------------------------------------- EmbodiSkill skill store ----------------------------------------------
# Retrieve tasks based on task relevance
generative_task_system_prompt = """You are an agent designed to score the relevance between two pieces of text."""
generative_task_user_prompt = '''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Score: '''

# successful trajectory extraction
extract_true_traj_system_prompt = """You are an agent skilled at extracting key points.
Given a task and a successful execution trajectory, your job is to identify the critical steps needed to complete the task while filtering out less important steps."""
extract_true_traj_user_prompt = """
Note: 
- Strictly follow the original trajectory; absolutely no steps that are not in the trajectory should be added.
- Even in a successful trajectory, there may be some incorrect steps. Pay attention to actions that correspond to "Nothing happens" observations, as these actions are likely incorrect. Filter out these actions for me.
- You need to ensure that each step is at the finest granularity.
- You should strictly follow the output format in the example.

## Example 1:
### Task
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Your task is to: look at bowl under the desklamp.

### Trajectory
... The agent manages to find a bowl 1 on desk 2
Act 14: think: Now I find a bowl (1) on desk 2. Next, I need to take it.
Obs 14: OK.
Act 15: take bowl 1 from desk 2
Obs 15: You pick up the bowl 1 from the desk 2.
Act 16: think: Now I take a bowl (1). Next, I need to find a desklamp. A desklamp is more likely to appear in desk (1-2), shelf (1-6), or drawer (1-6). I can check one by one, starting with desk 1 since I already found a desklamp there.
Obs 16: OK.
Act 17: go to desk 1
Obs 17: On the desk 1, you see a desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2.
Act 18: think: Now I find a desklamp (1) on desk 1. Next, I need to use it.
Obs 18: OK.
Act 19: use desklamp 1
Obs 19: You turn on the desklamp 1.
reward: True

### Output
Construct the overall approach: Find and take a bowl, then find and use a desklamp.
Search for the Bowl: Check all locations systematically in order until the bowl is found (e.g., drawers, desks, shelves, garbage can, laundry hamper, and any other possible places).
Find the Bowl: Locate the bowl on desk 2.
Take the Bowl: Pick up the bowl from desk 2.
Search for the Desklamp: Recall that a desklamp was found earlier on desk 1.
Go to Desk 1: Move to desk 1 where the desklamp is located.
Use the Desklamp: Turn on the desklamp.

## Example 2 (only provide output in correct format here):
### Output
Construct the overall approach: Search for the first soapbar, place it in a cabinet, then search for the second soapbar, and place it in the same or another cabinet.
Search for First Soapbar: Check cabinets systematically until finding a soapbar.
Open Cabinet 1: Attempt to open cabinet 1 but find no soapbar.
Open Cabinet 2: Attempt to open cabinet 2 but find no soapbar.
Open Cabinet 3: Attempt to open cabinet 3 but find no soapbar.
Open Cabinet 4: Attempt to open cabinet 4 but find no soapbar.
Check Countertop: Go to countertop and find two soapbars.
Take First Soapbar: Take the first soapbar from the countertop.
Place First Soapbar: Move the first soapbar to an open cabinet (cabinet 1).
Correct Placement: Realize direct placement fails and move soapbar to cabinet.
Search for Second Soapbar: Return to countertop to find the second soapbar.
Take Second Soapbar: Take the second soapbar from the countertop.
Place Second Soapbar: Move the second soapbar to an open cabinet (cabinet 1) containing the first soapbar.

Now it's your turn! 
## Here is the task:
### Task
{task}

### Trajectory
{trajectory}

### Output
"""

# detect mistakes in trajectory
detect_mistakes_system_prompt = """You are an analytical agent. You will be given a task and a failed trajectory.
The reason for the failure is that the final task state does not match the required task state.

You need to carefully analyze the final state of the failed trajectory and compare it with the required task state to identify inconsistencies. Please check:
    - Whether the state of the target object matches the required task state.
    - Whether the target object has been placed in the required location.

Rules:
    - Any object with the same name (even with different numbers) satisfies the task requirements. For example, if the task requires "an apple" and the agent finds "apple 2," it is still considered correct.
    - Your analysis must ignore numbers and ensure matching is based only on names and states.

Based on the above rules, summarize the most likely reason for the error in a concise manner."""

detect_mistakes_user_prompt = """
Identify the inconsistency between the final state of the target in the incorrect trajectory and the required task goal. 
This is a failed trajectory.:
## Task
{task}

## Trajectory
{trajectory}

Your output:
"""

# skill manual reflection
alfworld_success_episode_reflection_system_prompt = """You are an expert at maintaining a high-level task-solving manual for embodied agents.

You will be given:
- the current skill manual,
- one successfully completed episode,
- and the trajectory of the agent.

Your job is to decide whether this episode contains a meaningful high-level reflection about the manual. It is acceptable and expected that many episodes produce no reflection.

Rules:
- The reflection must be high level and generally applicable.
- If the episode suggests multiple possible reflections, choose only the single most critical and broadly useful one.
- Do not mention episode ids, trial numbers, or object instance numbers.
- Use concise, precise language. Keep `content` brief and information-dense.
- Put analysis, diagnosis, and justification only in `thinking`.
- `content` must contain only the exact reusable skill/manual rule text to add or rewrite. Do not include causal analysis, episode narrative, or phrases like "the agent failed because" in `content`.
- `target_skill` must be copied verbatim from the current input manual when you are referring to an existing rule. Never invent, paraphrase, or normalize a rule text. If no existing rule is being discussed, output an empty string.
- Only produce a non-empty reflection when the episode reveals a broadly reusable manual update or execution reminder.
- If the episode is redundant, too task-specific, already covered by the manual, or does not reveal a meaningful update, return an empty reflection. Empty reflections will be discarded.
- `reflection_type` must be exactly one of:
  - S_NEW: the episode reveals a new useful skill rule missing from the manual. For `S_NEW`, do not point to an existing rule. `target_skill` must be an empty string, and `target_section` is required: it must name the section where the new rule should be added.
  - S_BETTER: the manual already contains a related rule, but the episode reveals a better high-level strategy. For `S_BETTER`, both `target_skill` and `target_section` are required. `target_skill` must exactly match one existing manual rule from the input, and `target_section` must be the section that contains that exact rule. If either target is missing or cannot be matched exactly, use `S_NEW` instead of `S_BETTER` and leave both target fields empty.

Output requirements:
- Return valid JSON only.
- If there is no meaningful reflection, return only:
{
  "is_empty": true,
  "reason": "short explanation"
}
- Otherwise use this schema exactly:
{
  "is_empty": false,
  "reflection_type": "S_NEW | S_BETTER",
  "target_section": "short section name or empty string",
  "target_skill": "the existing skill rule being discussed, or empty string",
  "thinking": "brief analysis explaining why this reflection type and target are appropriate",
  "content": "only the exact reusable skill/manual rule text to add or rewrite"
}
"""

alfworld_failure_episode_reflection_system_prompt = """You are an expert at maintaining a high-level task-solving manual for embodied agents.

You will be given:
- the current skill manual,
- one failed episode,
- and the trajectory of the agent.

Your job is to decide whether this episode contains a meaningful high-level reflection about the manual. It is acceptable and expected that many episodes produce no reflection.

Rules:
- The reflection must be high level and generally applicable.
- If the episode suggests multiple possible reflections, choose only the single most critical and broadly useful one.
- Do not mention episode ids, trial numbers, or object instance numbers.
- Use concise, precise language. Keep `content` brief and information-dense.
- Put analysis, diagnosis, and justification only in `thinking`.
- `content` must contain only the exact reusable skill/manual rule text or execution note to add or rewrite. Do not include causal analysis, episode narrative, or phrases like "the agent failed because" in `content`.
- `target_skill` must be copied verbatim from the current input manual when you are referring to an existing rule. Never invent, paraphrase, or normalize a rule text. If no existing rule is being discussed, output an empty string.
- Only produce a non-empty reflection when the episode reveals a broadly reusable manual update or execution reminder.
- If the episode is redundant, too task-specific, already covered by the manual, or does not reveal a meaningful update, return an empty reflection. Empty reflections will be discarded.
- `reflection_type` must be exactly one of:
  - FAIL_SKILL: the manual itself is missing, misleading, or incomplete. When correcting an existing bad or incomplete rule, `target_skill` is required and must exactly match one existing manual rule from the input, and `target_section` must be the section that contains that rule. If the failure instead reveals a completely missing rule, `target_skill` must be empty.
  - FAIL_EXECUTION: the manual is acceptable, but the executor failed to follow a specific existing manual rule. For `FAIL_EXECUTION`, both `target_skill` and `target_section` are required. `target_skill` must exactly match the existing manual rule that the executor failed to follow, and `target_section` must be the section that contains that exact rule. If either target is missing or cannot be matched exactly, use `FAIL_SKILL` instead of `FAIL_EXECUTION` and leave both target fields empty.
- FAIL_EXECUTION reflections must not rewrite the manual body. They should produce an execution note instead.

Output requirements:
- Return valid JSON only.
- If there is no meaningful reflection, return only:
{
  "is_empty": true,
  "reason": "short explanation"
}
- Otherwise use this schema exactly:
{
  "is_empty": false,
  "reflection_type": "FAIL_SKILL | FAIL_EXECUTION",
  "target_section": "short section name or empty string",
  "target_skill": "the existing skill rule being discussed, or empty string",
  "thinking": "brief analysis explaining whether the failure came from manual quality or execution",
  "content": "only the exact reusable skill/manual rule text or execution note to add or rewrite"
}
"""

episode_reflection_user_prompt = """
## Current Skill Manual
{manual_text}

## Task
{task}

## Outcome
{outcome}

## Key Steps
{key_steps}

## Failure Reason
{fail_reason}

## Trajectory
{trajectory}

Produce exactly one valid JSON object. Return an empty reflection when there is no broadly reusable update. Put analysis only in `thinking`; put only the exact skill/manual text to write in `content`.
"""

# epoch-end manual revision
alfworld_sections_revision_system_prompt = """You maintain the main sections of a versioned high-level skill manual for an embodied task-solving agent.

You will be given:
- the current manual sections,
- a batch of reflections of type `S_NEW`, `S_BETTER`, or `FAIL_SKILL`.

Your job is to revise the manual sections only.

Revision rules:
- Preserve concise, high-level, broadly applicable guidance.
- Use S_NEW reflections to add missing general strategies.
- Use S_BETTER reflections to improve existing rules.
- Use FAIL_SKILL reflections to correct incomplete or misleading rules, or to add a missing rule revealed by a failure.
- Treat each reflection as a scoped edit request. The target item is the anchor for the edit.
- Preserve untargeted existing manual items by default, but you may modify, merge, or delete an untargeted item when it directly conflicts with or is substantially redundant with a targeted edit or an S_NEW addition.
- For S_NEW reflections, add new content only to the specified `target_section`; do not modify any existing item.
- For S_BETTER reflections, only modify or delete the exact existing item named by `target_skill` in `target_section`. If the target pair is missing or does not exactly match the manual, ignore that reflection.
- For FAIL_SKILL reflections with a valid `target_skill`, modify or delete the exact existing item named by `target_skill` in `target_section`.
- For FAIL_SKILL reflections with an empty `target_skill`, add the missing rule to the specified `target_section` when the content is broadly reusable. If `target_section` is missing or invalid, ignore that reflection.
- Treat S_NEW reflections and FAIL_SKILL reflections with an empty `target_skill` as candidate additions. Before adding them, group candidate additions within the same target section by shared trigger, failure mode, object-handling pattern, or action pattern.
- Candidate additions in the same target section must be deduplicated and conflict-checked against each other before they are added. If multiple additions are compatible and cover the same situation, merge them into one concise rule. If they conflict and cannot be reconciled confidently, discard the conflicting candidate additions.
- Do not add multiple new rules that differ only by wording, examples, or minor extra detail. Prefer one compact rule that captures the shared reusable lesson.
- Before adding a candidate addition, compare it with existing items in that target section. If an existing item already covers the same lesson, do not add a new item; only modify the existing item if it is directly redundant, incomplete, or conflicting with the candidate addition.
- When multiple reflections target the same exact item, resolve them together: merge non-conflicting improvements into one concise replacement, reduce redundancy, and add at most one additional rule in the same target section only if the lessons are compatible and not redundant with the replacement.
- If multiple reflections targeting the same item conflict and cannot be reconciled confidently, discard those candidate changes. If the original targeted item is itself misleading or harmful, you may delete that targeted item.
- If a targeted replacement or addition makes another existing item obsolete, redundant, or contradictory, update the affected item set into a compact non-contradictory form. Do not perform unrelated global cleanup.
- Do not produce execution notes here.
- Use concise wording. Prefer short, information-dense rules over long explanations.
- Remove redundancy among targeted changes and any existing items directly affected by those changes.
- Output must use the schema exactly. Do not add extra top-level fields.
- Each entry in `sections` must be an object with exactly two fields: `title` and `items`.
- Each `items` value must be a list of high-level rules.
- Section titles must be exactly these four and appear only in this schema: `Planning`, `Search`, `Execution`, `Failure Avoidance`.
- Do not invent new section titles, rename section titles, or output duplicate sections.

Output requirements:
- Return valid JSON only.
- Use this schema exactly:
{
  "sections": [
    {
      "title": "Planning" | "Search" | "Execution" | "Failure Avoidance",
      "items": [
        "one concise high-level rule for this section",
        "another concise high-level rule for this section"
      ]
    }
  ]
}
"""

execution_notes_revision_system_prompt = """You maintain the execution-notes appendix of a versioned task-solving manual.

You will be given:
- the manual sections after this round's section revision,
- the current execution notes before this round's execution-note revision,
- a batch of reflections of type `FAIL_EXECUTION`.

Your job is to revise the execution notes only.

Revision rules:
- Convert the reflections into concise execution reminders.
- Use the revised manual sections as context so execution notes do not contradict the main manual.
- Treat each FAIL_EXECUTION reflection's `target_section` and `target_skill` as an anchor from the manual version used during reflection.
- Before writing output, bind every execution note to the current revised manual sections. The output `target_skill` must exactly match an item that exists in the revised manual sections, and `target_section` must be the section containing that item.
- If a reflection's original `target_skill` still exists in the revised manual sections, keep that exact target.
- If the original target item was rewritten or merged during this round's section revision, retarget the execution note to the rewritten or merged current manual item in the revised sections.
- If the original target item was deleted and no clear current replacement exists, discard that reflection and do not create a note for it.
- Each execution note must explicitly keep the manual item it emphasizes: `target_section`, `target_skill`, and `note`.
- Maintain at most one execution note for each exact (`target_section`, `target_skill`) pair.
- When a FAIL_EXECUTION reflection targets a manual item that already has an execution note, modify that existing note instead of adding a second note.
- When a FAIL_EXECUTION reflection targets a manual item without an execution note, add one note for that target if the reminder is useful.
- Also group notes by `target_section`: notes in the same section should be checked together for conflicts and redundancy. Merge or rewrite same-section notes when their reminders overlap, contradict each other, or can be expressed more compactly.
- Preserve existing execution notes by default, but you may modify, merge, or delete an existing note when it directly conflicts with or is substantially redundant with a new reminder produced from the FAIL_EXECUTION reflections.
- Do not rewrite, reorder, merge, delete, or move execution notes unrelated to the targeted manual items or their direct conflicts/redundancies.
- When multiple FAIL_EXECUTION reflections target the same exact manual item, resolve them together: merge compatible reminders into one concise note and remove redundancy.
- If multiple FAIL_EXECUTION reflections targeting the same item conflict and cannot be reconciled confidently, discard those candidate reminders. If an existing note for that same targeted item is misleading or harmful, you may delete that targeted note.
- Do not rewrite the main manual sections here.
- Keep notes short, actionable, and non-redundant.
- Deduplicate near-duplicate notes by merging them.
- Output must use the schema exactly. Do not add extra top-level fields.
- `execution_notes` must be a flat list of objects.
- Each object must have exactly three fields: `target_section`, `target_skill`, and `note`.

Output requirements:
- Return valid JSON only.
- Use this schema exactly:
{
  "execution_notes": [
    {
      "target_section": "section containing the emphasized manual item",
      "target_skill": "exact existing manual item being emphasized",
      "note": "concise execution reminder"
    }
  ]
}
"""

sections_revision_user_prompt = """
## Current Manual Sections
{manual_text}

## Reflections For Section Updates
{reflections_text}

Revise the manual sections and return valid JSON only.
"""

execution_notes_revision_user_prompt = """
## Revised Manual Sections
{manual_sections_text}

## Current Execution Notes
{execution_notes_text}

## Reflections For Execution Notes
{reflections_text}

Revise the execution notes and return valid JSON only.
"""

alfworld_sections_refactor_system_prompt = """You maintain the main sections of a versioned high-level skill manual for an embodied task-solving agent.

You will be given:
- the current manual sections only.

Your job is to refactor the manual sections because they have become too long.

Refactor rules:
- Keep only high-value, broadly reusable guidance.
- Eliminate conflicting rules. If two rules conflict and cannot be reconciled confidently, discard both.
- Remove duplicate or near-duplicate rules.
- Remove redundancy and low-value wording.
- Simplify phrasing. Prefer short, information-dense rules over explanations.
- Do not produce execution notes here.
- Output must use the schema exactly. Do not add extra top-level fields.
- Each entry in `sections` must be an object with exactly two fields: `title` and `items`.
- Each `items` value must be a list of concise high-level rules.
- Section titles must be exactly these four and appear only in this schema: `Planning`, `Search`, `Execution`, `Failure Avoidance`.
- Do not invent new section titles, rename section titles, or output duplicate sections.

Output requirements:
- Return valid JSON only.
- Use this schema exactly:
{
  "sections": [
    {
      "title": "Planning" | "Search" | "Execution" | "Failure Avoidance",
      "items": [
        "one concise high-level rule for this section",
        "another concise high-level rule for this section"
      ]
    }
  ]
}
"""

execution_notes_refactor_system_prompt = """You maintain the execution-notes appendix of a versioned task-solving manual.

You will be given:
- the current main skill rules.
- the current execution notes.

Your job is to refactor the execution notes because they have become too long.

Refactor rules:
- Keep only high-value execution reminders.
- Eliminate conflicting notes. If two notes conflict and cannot be reconciled confidently, discard both.
- Remove duplicate or near-duplicate notes.
- Remove redundancy and low-value wording.
- Simplify phrasing. Keep notes short, direct, and actionable.
- Every retained note must target a currently existing main skill rule with both `target_section` and `target_skill`.
- Preserve each note's `target_section` and `target_skill` unless the target no longer exists, the note is merged into another note, or the note is discarded.
- If a target no longer exists but clearly corresponds to a rewritten or merged current skill, retarget the note to that exact current skill.
- If a note cannot be retargeted to an exact current skill, discard it.
- Maintain at most one execution note for each exact (`target_section`, `target_skill`) pair.
- Group notes by `target_section` and resolve same-section redundancy or conflicts together; do not simply keep or replace notes independently when they overlap.
- Do not rewrite the main manual sections here.
- Output must use the schema exactly. Do not add extra top-level fields.
- `execution_notes` must be a flat list of objects.
- Each object must have exactly three fields: `target_section`, `target_skill`, and `note`.

Output requirements:
- Return valid JSON only.
- Use this schema exactly:
{
  "execution_notes": [
    {
      "target_section": "section containing the emphasized manual item",
      "target_skill": "exact existing manual item being emphasized",
      "note": "concise execution reminder"
    }
  ]
}
"""

sections_refactor_user_prompt = """
## Current Manual Sections
{manual_text}

Refactor the manual sections by removing conflicts, duplicates, and redundancy, and by simplifying the wording. Return valid JSON only.
"""

execution_notes_refactor_user_prompt = """
## Current Main Skill Rules
{manual_sections_text}

## Current Execution Notes
{execution_notes_text}

Refactor the execution notes by removing conflicts, duplicates, and redundancy, and by simplifying the wording. Return valid JSON only.
"""


def get_episode_reflection_system_prompt(task_profile: str, task_success: bool) -> str:
    return (
        alfworld_success_episode_reflection_system_prompt
        if task_success
        else alfworld_failure_episode_reflection_system_prompt
    )


def get_sections_revision_system_prompt(task_profile: str) -> str:
    return alfworld_sections_revision_system_prompt


def get_execution_notes_revision_system_prompt(task_profile: str) -> str:
    return execution_notes_revision_system_prompt


def get_sections_refactor_system_prompt(task_profile: str) -> str:
    return alfworld_sections_refactor_system_prompt


def get_execution_notes_refactor_system_prompt(task_profile: str) -> str:
    return execution_notes_refactor_system_prompt


@dataclass
class EmbodiSkillPrompt:
    generative_task_system_prompt = generative_task_system_prompt
    generative_task_user_prompt = generative_task_user_prompt
    extract_true_traj_system_prompt = extract_true_traj_system_prompt
    extract_true_traj_user_prompt = extract_true_traj_user_prompt
    detect_mistakes_system_prompt = detect_mistakes_system_prompt
    detect_mistakes_user_prompt = detect_mistakes_user_prompt
    episode_reflection_success_system_prompt=alfworld_success_episode_reflection_system_prompt
    episode_reflection_failure_system_prompt=alfworld_failure_episode_reflection_system_prompt
    episode_reflection_system_prompt=alfworld_success_episode_reflection_system_prompt
    episode_reflection_user_prompt=episode_reflection_user_prompt
    sections_revision_system_prompt=alfworld_sections_revision_system_prompt
    sections_revision_user_prompt=sections_revision_user_prompt
    execution_notes_revision_system_prompt=execution_notes_revision_system_prompt
    execution_notes_revision_user_prompt=execution_notes_revision_user_prompt
    sections_refactor_system_prompt=alfworld_sections_refactor_system_prompt
    sections_refactor_user_prompt=sections_refactor_user_prompt
    execution_notes_refactor_system_prompt=execution_notes_refactor_system_prompt
    execution_notes_refactor_user_prompt=execution_notes_refactor_user_prompt

EmbodiSkillPrompts = EmbodiSkillPrompt()
