from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import os
import copy
import json
import re
from typing import Iterable, Any, Optional, Callable
import networkx as nx
import numpy as np
from finch import FINCH
import pickle
from datetime import datetime, timezone

from .skill_base import MASSkillBase
from ..common import MASMessage, StateChain
from ..utils import cosine_similarity
from .prompt import (
    EmbodiSkillPrompts,
    get_episode_reflection_system_prompt,
    get_execution_notes_refactor_system_prompt,
    get_execution_notes_revision_system_prompt,
    get_manual_display_name,
    get_sections_refactor_system_prompt,
    get_sections_revision_system_prompt,
    get_manual_section_titles,
    get_manual_section_templates,
    normalize_manual_profile,
)
from agentkit.utils import load_json, write_json
from agentkit.llm import Message

@dataclass
class EmbodiSkill(MASSkillBase):
    """
    EmbodiSkill
    A three-tier hierarchical graph structure composed of the Skill Graph,
    Query Graph, and Interaction Graph.

    1. Interaction Graph - Trajectory Condensation: During the task-solving process, the multi-agent system (MAS) generates a chain of states, where each state represents a step in the process of arriving at the final answer. Behind each state is a corresponding message graph.
    Each task corresponds to a chain of states, which connects the middle and bottom layers of the multi-layer graph.
    2. Query Graph - Based on the current task, the system retrieves previously successful records. A k-hop approach is used to expand the search scope within the query graph.
    3. Skill Graph - Skill Retrieval: Relevant skills are retrieved based on the
    current task to assist in decision-making.
    """
    def __post_init__(self):
        super().__post_init__()
        
        self.main_skill = Chroma(          
            embedding_function=self.embedding_func,  
            persist_directory=self.persist_dir         
        )

        self._hop: int = self.global_config.get('hop', 1)
        self._llm_concurrency: int = max(1, int(self.global_config.get('llm_concurrency', 4)))
        self._current_epoch_id: Optional[int] = self.global_config.get('current_epoch_id')
        self._task_profile: str = normalize_manual_profile(
            self.global_config.get('task') or self.global_config.get('task_name')
        )
        self._manual_sections_refactor_threshold: int = max(
            1,
            int(self.global_config.get('manual_sections_refactor_threshold', 100)),
        )
        self._execution_notes_refactor_threshold: int = max(
            1,
            int(self.global_config.get('execution_notes_refactor_threshold', 50)),
        )
        self._manual_reflection_max_tokens: int = max(
            1,
            int(self.global_config.get('manual_reflection_max_tokens', 512)),
        )
        self._manual_revision_max_tokens: int = max(
            1,
            int(self.global_config.get('manual_revision_max_tokens', 2048)),
        )
        self.global_config['task_profile'] = self._task_profile
        self.global_config.setdefault('project_name', 'EmbodiSkill')

        self.skill_cache: list[str] = []
        self.current_manual_context: dict[str, Any] | None = None
        self.task_dir: str = os.path.join(self.persist_dir, 'task')
        self.manual_dir: str = os.path.join(self.persist_dir, 'skill')
        self.manual_versions_dir: str = os.path.join(self.manual_dir, 'versions')
        self.reflections_dir: str = os.path.join(self.persist_dir, 'reflections')
        self.manual_file: str = os.path.join(self.manual_dir, 'current.json')
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.manual_dir, exist_ok=True)
        os.makedirs(self.manual_versions_dir, exist_ok=True)
        os.makedirs(self.reflections_dir, exist_ok=True)

        self.task_layer = TaskLayer(
            working_dir=self.task_dir,
            namespace='task_layer',
            task_storage=self.main_skill
        )
        self.manual_state: dict[str, Any] = self._load_or_init_manual_state()
        self.reflections_file: str = self._reflection_epoch_file(self._current_epoch_id)
        self.reflections_skill: list[dict[str, Any]] = self._load_reflections_for_epoch(self._current_epoch_id)

        print(self._get_hyperparams_dict())
    
    def _get_hyperparams_dict(self) -> dict:
        return {
            'hop': self._hop,
            'llm_concurrency': self._llm_concurrency,
            'skill_manual': 'manual_reflection',
            'current_epoch_id': self._current_epoch_id,
            'task_profile': self._task_profile,
            'manual_sections_refactor_threshold': self._manual_sections_refactor_threshold,
            'execution_notes_refactor_threshold': self._execution_notes_refactor_threshold,
            'manual_reflection_max_tokens': self._manual_reflection_max_tokens,
            'manual_revision_max_tokens': self._manual_revision_max_tokens,
            'project_name': self.global_config.get('project_name', 'EmbodiSkill'),
            'working_dir': self.persist_dir
        }

    def _reflection_epoch_label(self, epoch_id: Optional[int]) -> str:
        if epoch_id is None:
            return 'unknown'
        if isinstance(epoch_id, int):
            return f'{epoch_id:03d}'
        label = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(epoch_id).strip()).strip('_')
        if label.isdigit():
            return f'{int(label):03d}'
        return label or 'unknown'

    def _reflection_epoch_file(self, epoch_id: Optional[int]) -> str:
        return os.path.join(self.reflections_dir, f'epoch_{self._reflection_epoch_label(epoch_id)}.json')

    def _reflection_belongs_to_epoch(self, reflection: dict[str, Any], epoch_id: Optional[int]) -> bool:
        saved_epoch_id = reflection.get('epoch_id')
        if epoch_id is None:
            return saved_epoch_id is None
        return str(saved_epoch_id) == str(epoch_id)

    def _load_reflections_for_epoch(self, epoch_id: Optional[int]) -> list[dict[str, Any]]:
        epoch_file = self._reflection_epoch_file(epoch_id)
        self.reflections_file = epoch_file

        loaded = load_json(epoch_file)
        return loaded if isinstance(loaded, list) else []

    def _set_current_epoch_id(self, epoch_id: Optional[int]) -> None:
        if str(epoch_id) == str(self._current_epoch_id) and hasattr(self, 'reflections_skill'):
            return
        self._current_epoch_id = epoch_id
        if hasattr(self, 'reflections_dir'):
            self.reflections_skill = self._load_reflections_for_epoch(epoch_id)

    def uses_fixed_skill_manual(self) -> bool:
        return True

    def _canonicalize_dedupe_key(self, text: str) -> str:
        normalized = re.sub(r'\s+', ' ', str(text or '').strip())
        normalized = normalized.rstrip(' .;:,_-')
        return normalized.casefold()

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen = set()
        deduped: list[str] = []
        for item in items:
            text = re.sub(r'\s+', ' ', str(item).strip())
            if not text:
                continue
            key = self._canonicalize_dedupe_key(text)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped

    def _normalize_execution_note(self, note: Any) -> Optional[dict[str, str]]:
        if not isinstance(note, dict):
            return None

        note_text = str(note.get('note', '')).strip()
        target_section = str(note.get('target_section', '')).strip()
        target_skill = str(note.get('target_skill', '')).strip()

        if not note_text or not target_section or not target_skill:
            return None
        return {
            'target_section': target_section,
            'target_skill': target_skill,
            'note': re.sub(r'\s+', ' ', note_text),
        }

    def _execution_note_key(self, note: dict[str, str]) -> str:
        target_section = str(note.get('target_section', '')).strip()
        target_skill = str(note.get('target_skill', '')).strip()
        return self._canonicalize_dedupe_key(f"{target_section}\n{target_skill}")

    def _section_skill_to_section_map(self, sections: list[dict[str, Any]]) -> dict[str, str]:
        skill_map: dict[str, str] = {}
        for section in sections:
            if not isinstance(section, dict):
                continue
            title = str(section.get('title', '')).strip()
            for item in section.get('items', []):
                item_text = str(item).strip()
                if item_text and item_text not in skill_map:
                    skill_map[item_text] = title
        return skill_map

    def _normalize_execution_notes(
        self,
        notes: Any,
        skill_to_section: Optional[dict[str, str]] = None,
    ) -> list[dict[str, str]]:
        raw_notes = notes if isinstance(notes, list) else []
        note_index_by_key: dict[str, int] = {}
        normalized_notes: list[dict[str, str]] = []
        for raw_note in raw_notes:
            note = self._normalize_execution_note(raw_note)
            if note is None:
                continue
            if skill_to_section is not None:
                matched_section = skill_to_section.get(note['target_skill'])
                if matched_section is None:
                    continue
                note['target_section'] = matched_section
            key = self._execution_note_key(note)
            if not key:
                continue
            if key in note_index_by_key:
                normalized_notes[note_index_by_key[key]] = note
                continue
            note_index_by_key[key] = len(normalized_notes)
            normalized_notes.append(note)
        return normalized_notes

    def _is_execution_notes_section_title(self, title: str) -> bool:
        normalized = re.sub(r'[^a-z]+', ' ', str(title or '').strip().lower()).strip()
        return normalized in {
            'execution notes',
            'execution note',
            'execution notes appendix',
            'notes appendix',
            'appendix',
        }

    def _normalize_section_title_key(self, title: str) -> str:
        return re.sub(r'[^a-z]+', ' ', str(title or '').strip().lower()).strip()

    def _allowed_section_title_map(self, profile: Optional[str] = None) -> dict[str, str]:
        titles = get_manual_section_titles(profile or self._task_profile)
        return {
            self._normalize_section_title_key(title): title
            for title in titles
        }

    def _manual_skill_to_section_map(self) -> dict[str, str]:
        return self._section_skill_to_section_map(self.manual_state.get('sections', []))

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _create_empty_manual_state(self, profile: Optional[str] = None) -> dict[str, Any]:
        manual_profile = normalize_manual_profile(profile or self._task_profile)
        return {
            'profile': manual_profile,
            'version': 1,
            'updated_at_utc': self._utc_now(),
            'sections': get_manual_section_templates(manual_profile),
            'execution_notes': [],
        }

    def _bootstrap_manual_state(self) -> dict[str, Any]:
        return self._create_empty_manual_state()

    def _normalize_manual_state(self, state: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(state, dict):
            return self._create_empty_manual_state()

        normalized = self._create_empty_manual_state(state.get('profile'))
        normalized['profile'] = normalize_manual_profile(
            state.get('profile') or self._task_profile
        )
        normalized['version'] = int(state.get('version', normalized['version']) or 1)
        normalized['updated_at_utc'] = state.get('updated_at_utc') or self._utc_now()

        allowed_title_map = self._allowed_section_title_map(normalized['profile'])
        normalized['sections'] = get_manual_section_templates(normalized['profile'])
        section_items_map = {section['title']: [] for section in normalized['sections']}

        sections = state.get('sections', [])
        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue
                title = str(section.get('title', '')).strip()
                items = [str(item).strip() for item in section.get('items', []) if str(item).strip()]
                if self._is_execution_notes_section_title(title):
                    continue
                canonical_title = allowed_title_map.get(self._normalize_section_title_key(title))
                if canonical_title is None:
                    continue
                section_items_map[canonical_title].extend(items)

        normalized['sections'] = [
            {
                'title': section['title'],
                'items': self._dedupe_preserve_order(section_items_map.get(section['title'], [])),
            }
            for section in normalized['sections']
        ]

        normalized['execution_notes'] = self._normalize_execution_notes(
            state.get('execution_notes', []),
            skill_to_section=self._section_skill_to_section_map(normalized['sections']),
        )

        return normalized

    def _manual_state_is_effectively_empty(self) -> bool:
        if any(self.manual_state.get('execution_notes', [])):
            return False
        for section in self.manual_state.get('sections', []):
            if any(str(item).strip() for item in section.get('items', [])):
                return False
        return True

    def _set_task_profile(self, task_name: Optional[str]) -> None:
        profile = normalize_manual_profile(task_name)
        if profile == 'default' or profile == self._task_profile:
            return
        self._task_profile = profile
        self.global_config['task_profile'] = profile
        if hasattr(self, 'manual_state') and isinstance(self.manual_state, dict):
            current_profile = normalize_manual_profile(self.manual_state.get('profile'))
            if current_profile != profile:
                if self._manual_state_is_effectively_empty():
                    self.manual_state = self._create_empty_manual_state(profile)
                    self._save_manual_state()
                else:
                    self.manual_state['profile'] = profile

    def init_task_context(
        self,
        task_main: str,
        task_description: str = None,
    ) -> MASMessage:
        current_epoch_id = self.global_config.get('current_epoch_id')
        if current_epoch_id is not None:
            self._set_current_epoch_id(current_epoch_id)
        self._set_task_profile(
            self.global_config.get('task')
            or self.global_config.get('task_name')
            or task_main
            or task_description
        )
        return super().init_task_context(task_main, task_description)

    def _save_manual_state(self) -> None:
        self.manual_state['updated_at_utc'] = self._utc_now()
        write_json(self.manual_state, self.manual_file)
        self._save_manual_version_snapshot(self.manual_state)

    def _save_reflections(self) -> None:
        self.reflections_file = self._reflection_epoch_file(self._current_epoch_id)
        write_json(self.reflections_skill, self.reflections_file)

    def _manual_version_file(self, version: int) -> str:
        return os.path.join(self.manual_versions_dir, f'skill_v{version}.json')

    def _save_manual_version_snapshot(self, manual_state: dict[str, Any]) -> None:
        version = int(manual_state.get('version', 1))
        snapshot = copy.deepcopy(manual_state)
        snapshot.setdefault('saved_at_utc', self._utc_now())
        write_json(snapshot, self._manual_version_file(version))

    def _load_or_init_manual_state(self) -> dict[str, Any]:
        loaded = load_json(self.manual_file)
        if loaded is not None:
            state = self._normalize_manual_state(loaded)
            if state != loaded:
                write_json(state, self.manual_file)
            version_file = self._manual_version_file(int(state.get('version', 1)))
            if not os.path.exists(version_file):
                self._save_manual_version_snapshot(state)
        else:
            state = self._bootstrap_manual_state()
            write_json(state, self.manual_file)
            self._save_manual_version_snapshot(state)
        return state

    def _manual_lines(self, include_execution_notes: bool = True) -> list[str]:
        lines: list[str] = []
        for section in self.manual_state.get('sections', []):
            title = str(section.get('title', '')).strip()
            for item in section.get('items', []):
                item_text = str(item).strip()
                if not item_text:
                    continue
                lines.append(f"[{title}] {item_text}" if title else item_text)

        if include_execution_notes:
            for note in self._normalize_execution_notes(self.manual_state.get('execution_notes', [])):
                note_text = str(note.get('note', '')).strip()
                target_section = str(note.get('target_section', '')).strip()
                target_skill = str(note.get('target_skill', '')).strip()
                if not note_text:
                    continue
                if target_skill:
                    target_label = f"{target_section}: {target_skill}" if target_section else target_skill
                    lines.append(f"[Execution Note for {target_label}] {note_text}")
                else:
                    lines.append(f"[Execution Note] {note_text}")
        return lines

    def get_active_manual_prompt_lines(self) -> list[str]:
        return self._manual_lines(include_execution_notes=True)

    def get_active_manual_data(self) -> dict[str, Any]:
        skill_text = self.get_active_manual_text()
        prompt_lines = self.get_active_manual_prompt_lines()
        return {
            'profile': self.manual_state.get('profile', self._task_profile),
            'version': self.manual_state.get('version', 1),
            'manual_text': skill_text,
            'skill_text': skill_text,
            'prompt_lines': prompt_lines,
            'skill_lines': prompt_lines,
        }

    def get_active_manual_text(self) -> str:
        parts = [
            self._format_sections_text(
                profile=self.manual_state.get('profile', self._task_profile),
                version=int(self.manual_state.get('version', 1)),
                sections=self.manual_state.get('sections', []),
            )
        ]
        notes = self._format_execution_notes_text(self.manual_state.get('execution_notes', []))
        if notes != "[Execution Notes Appendix]\n(none)":
            parts.append(notes)
        return '\n'.join(parts).strip()

    def _format_sections_text(
        self,
        *,
        profile: str,
        version: int,
        sections: list[dict[str, Any]],
    ) -> str:
        parts = [f"{get_manual_display_name(profile)} v{version}"]
        for section in sections:
            title = str(section.get('title', '')).strip()
            items = [str(item).strip() for item in section.get('items', []) if str(item).strip()]
            if title:
                parts.append(f"\n[{title}]")
            for idx, item in enumerate(items, 1):
                parts.append(f"{idx}. {item}")
        return '\n'.join(parts).strip()

    def _format_execution_notes_text(self, execution_notes: list[Any]) -> str:
        notes = self._normalize_execution_notes(execution_notes)
        if not notes:
            return "[Execution Notes Appendix]\n(none)"

        parts = ["[Execution Notes Appendix]"]
        for idx, note in enumerate(notes, 1):
            target_section = str(note.get('target_section', '')).strip() or "(none)"
            target_skill = str(note.get('target_skill', '')).strip() or "(none)"
            note_text = str(note.get('note', '')).strip()
            parts.append(f"{idx}. Target Section: {target_section}")
            parts.append(f"   Target Skill: {target_skill}")
            parts.append(f"   Note: {note_text}")
        return '\n'.join(parts).strip()

    def get_active_manual_sections_text(self) -> str:
        return self._format_sections_text(
            profile=self.manual_state.get('profile', self._task_profile),
            version=int(self.manual_state.get('version', 1)),
            sections=self.manual_state.get('sections', []),
        )

    def get_active_execution_notes_text(self) -> str:
        return self._format_execution_notes_text(self.manual_state.get('execution_notes', []))

    def get_active_skill_prompt_lines(self) -> list[str]:
        return self.get_active_manual_prompt_lines()

    def get_active_skill_data(self) -> dict[str, Any]:
        return self.get_active_manual_data()

    def get_active_skill_text(self) -> str:
        return self.get_active_manual_text()

    def _count_section_items(self, sections: Optional[list[dict[str, Any]]] = None) -> int:
        active_sections = sections if sections is not None else self.manual_state.get('sections', [])
        count = 0
        for section in active_sections:
            if not isinstance(section, dict):
                continue
            count += len([item for item in section.get('items', []) if str(item).strip()])
        return count

    def _count_execution_notes(self, execution_notes: Optional[list[Any]] = None) -> int:
        active_notes = execution_notes if execution_notes is not None else self.manual_state.get('execution_notes', [])
        return len(self._normalize_execution_notes(active_notes))

    def set_episode_manual_context(self, manual_data: Optional[dict[str, Any]] = None) -> None:
        self.current_manual_context = manual_data or self.get_active_manual_data()

    def _next_reflection_id(self) -> int:
        return len(self.reflections_skill) + 1

    def _strip_code_fences(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        cleaned = self._strip_code_fences(response)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _derive_reflection_flags(
        self,
        reflection_type: str,
        task_success: bool,
        is_empty_reflection: bool,
    ) -> tuple[str, bool, bool]:
        if is_empty_reflection:
            return "", False, False

        normalized_type = str(reflection_type or "").strip().upper()
        allowed_types = {"S_NEW", "S_BETTER"} if task_success else {"FAIL_SKILL", "FAIL_EXECUTION"}
        fallback_type = "S_NEW" if task_success else "FAIL_SKILL"

        if normalized_type not in allowed_types:
            print(
                f"[EmbodiSkill] Invalid reflection_type={normalized_type!r} for "
                f"task_success={task_success}; fallback to {fallback_type}."
            )
            normalized_type = fallback_type

        should_modify_manual = normalized_type != "FAIL_EXECUTION"
        should_add_execution_note = normalized_type == "FAIL_EXECUTION"
        return normalized_type, should_modify_manual, should_add_execution_note

    def _validate_reflection_target(
        self,
        *,
        reflection_type: str,
        target_section: str,
        target_skill: str,
    ) -> tuple[str, str, str]:
        normalized_type = str(reflection_type or "").strip().upper()
        validated_target_section = str(target_section or "").strip()
        validated_target_skill = str(target_skill or "").strip()

        allowed_title_map = self._allowed_section_title_map(self.manual_state.get('profile', self._task_profile))
        if validated_target_section:
            canonical_title = allowed_title_map.get(self._normalize_section_title_key(validated_target_section))
            if canonical_title is None:
                print(
                    f"[EmbodiSkill] Invalid target_section={validated_target_section!r}; clearing it."
                )
                validated_target_section = ""
            else:
                validated_target_section = canonical_title

        skill_to_section = self._manual_skill_to_section_map()
        if validated_target_skill:
            matched_section = skill_to_section.get(validated_target_skill)
            if matched_section is None:
                print(
                    f"[EmbodiSkill] target_skill not found in current manual; clearing it. "
                    f"target_skill={validated_target_skill!r}"
                )
                validated_target_skill = ""
            else:
                validated_target_section = matched_section

        if normalized_type == "S_BETTER" and (not validated_target_section or not validated_target_skill):
            print(
                "[EmbodiSkill] S_BETTER reflection without a valid target_section and target_skill pair; "
                "downgrading to S_NEW."
            )
            normalized_type = "S_NEW"
            validated_target_section = ""
            validated_target_skill = ""

        if normalized_type == "FAIL_EXECUTION" and (not validated_target_section or not validated_target_skill):
            print(
                "[EmbodiSkill] FAIL_EXECUTION reflection without a valid target_section and target_skill pair; "
                "downgrading to FAIL_SKILL."
            )
            normalized_type = "FAIL_SKILL"
            validated_target_section = ""
            validated_target_skill = ""

        return normalized_type, validated_target_section, validated_target_skill

    def _parallel_map(self, items: list, fn):
        if len(items) <= 1 or self._llm_concurrency <= 1:
            return [fn(item) for item in items]

        with ThreadPoolExecutor(max_workers=min(self._llm_concurrency, len(items))) as executor:
            return list(executor.map(fn, items))


    def add_skill(self, mas_message: MASMessage) -> MASMessage:
        """
        Add the mas_message corresponding to a completed task into the skill store:
        Step 1: Sparsification - remove incorrect steps
        Step 2: Add the sparsified trajectories to the skill store
        Args:
            mas_message (MASMessage): The MAS message corresponding to a completed task

        Raises:
            ValueError: mas_message must have label!
        """
        # sparsification
        mas_message = self._extract_mas_message(mas_message=mas_message)  
        
        # add into the skill store
        self.task_layer.add_task_node(mas_message.task_main)

        meta_data: dict = MASMessage.to_dict(mas_message)
        skill_doc = Document(
            page_content=mas_message.task_main,   
            metadata=meta_data
        )
        if mas_message.label == True or mas_message.label == False:
            self.main_skill.add_documents([skill_doc])
        else:
            raise ValueError('The mas_message must have label!')
        
        self._index_done()
        return mas_message

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        if self.current_task_context is None:
            raise RuntimeError('The current inside-trial skill state is empty.')

        self.current_task_context.label = label
        if feedback is not None:
            self.current_task_context.task_description += f'\n- Environment feedback\n{feedback}\n'
        return self.add_skill(self.current_task_context)

    def _retrieve_skill_raw(
        self, 
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1, 
        threshold: float = 0.3
    ) -> tuple[list, list, list]:

        def sort_and_filter_by_similarity(docs: list[Document], threshold: float = 0.3) -> list[tuple[Document, float]]:
            result = []
            for doc in docs:
                embedding = self.embedding_func.embed_query(doc.page_content)
                sim = cosine_similarity(origin_embedding, embedding)
                if sim >= threshold:
                    result.append((doc, sim))

            result.sort(key=lambda x: x[1], reverse=True)
            return result

        true_tasks_doc: list[Document] = []
        false_tasks_doc: list[Document] = []
        
        # find related tasks in task layer
        related_point_num: int = max((successful_topk + failed_topk) // 2, 1)
        task_mains: list[str] = self.task_layer.retrieve_related_task(query_task=query_task, node_num=related_point_num, hop=self._hop)
        for task_main in task_mains:
            doc = self.main_skill.similarity_search(task_main, k=1)[0]

            if doc.metadata.get('label') == True:
                true_tasks_doc.append(doc)
            elif doc.metadata.get('label') == False:
                false_tasks_doc.append(doc)
            else:
                raise RuntimeError('The document object\'s metadata should have `label` attribute.')
        
        # If the specified number is not met, fill in the rest using similarity-based augmentation.
        if len(true_tasks_doc) < successful_topk:
            true_tasks_doc = self.main_skill.similarity_search(
                query=query_task, k=successful_topk, filter={'label': True}
            )
            for doc in true_tasks_doc:
                if doc not in true_tasks_doc:
                    true_tasks_doc.append(doc)
        
        if len(false_tasks_doc) < failed_topk:
            false_tasks_doc = self.main_skill.similarity_search(
                query=query_task, k=failed_topk, filter={'label': False}
            )
            for doc in false_tasks_doc:
                if doc not in false_tasks_doc:
                    false_tasks_doc.append(doc)

        # order by similarity        
        origin_embedding: list[float] = self.embedding_func.embed_query(query_task)
        true_tasks_doc_with_score = sort_and_filter_by_similarity(true_tasks_doc, threshold)[:successful_topk]
        false_tasks_doc_with_score = sort_and_filter_by_similarity(false_tasks_doc, threshold)[:failed_topk]

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc, _ in true_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc, _ in false_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)
        
        return true_task_messages, false_task_messages, []

    def retrieve_skill(
        self, 
        query_task: str,         
        successful_topk: int = 2, 
        failed_topk: int = 1,
        skill_topk: int = 10,
        threshold: float = 0.3,
    ) -> tuple[list, list, list]: 
        """Access the skill store and return the results.

        Args:
            query_task (str): The task to query.
            successful_topk (int, optional): Number of successful cases to retrieve. Defaults to 2.
            failed_topk (int, optional): Number of failed cases to retrieve. Defaults to 1.
            skill_topk (int, optional): Number of skills to retrieve. Defaults to 10.
            threshold (float, optional): Similarity threshold for retrieving cases. Defaults to 0.3.

        Returns:
            tuple[list, list, list]: A tuple containing successful cases, failed cases, and skills.
        """
        
        # retrieve raw tasks
        successful_task_trajectories: list[MASMessage]
        failed_task_trajectories: list[MASMessage]
        successful_task_trajectories, failed_task_trajectories, _ = self._retrieve_skill_raw(
            query_task, 2*successful_topk, 2*failed_topk, threshold)
        
        # Rerank retrieved successful trajectories with bounded parallelism.
        rerank_jobs: list[list[Message]] = []
        for success_task in successful_task_trajectories:
            prompt: str = EmbodiSkillPrompts.generative_task_user_prompt.format(
                trajectory=success_task.task_description + '\n' + success_task.task_trajectory,
                query_scenario=query_task
            )
            rerank_jobs.append([
                Message('system', EmbodiSkillPrompts.generative_task_system_prompt),
                Message('user', prompt),
            ])

        rerank_responses = self._parallel_map(rerank_jobs, lambda messages: self.llm_model(messages=messages))
        importance_score = [
            int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            for response in rerank_responses
        ]
        
        sorted_success_tasks = [task for _, task in sorted(zip(importance_score, successful_task_trajectories), 
                                                           key=lambda x: x[0], reverse=True)]
        top_success_task_trajectories = sorted_success_tasks[:successful_topk]
        # top_success_task_trajectories = successful_task_trajectories[:successful_topk]
        
        # directly get failed tasks
        top_fail_task_trajectories = failed_task_trajectories[:failed_topk]
        
        # Directly expose the current manual guidance.
        manual_lines = self.get_active_manual_prompt_lines()
        top_k_skills = manual_lines if skill_topk <= 0 else manual_lines[:skill_topk]
        self.skill_cache = top_k_skills
        self.skill_cache = self.skill_cache

        return top_success_task_trajectories, top_fail_task_trajectories, top_k_skills

    def reflect_episode(self, mas_message: MASMessage) -> Optional[dict[str, Any]]:
        manual_context = self.current_manual_context or self.get_active_manual_data()
        fail_reason = mas_message.get_extra_field('fail_reason') or ''
        key_steps = mas_message.get_extra_field('key_steps') or ''
        outcome = 'SUCCESS' if mas_message.label else 'FAILURE'
        user_prompt = EmbodiSkillPrompts.episode_reflection_user_prompt.format(
            manual_text=manual_context.get('manual_text', self.get_active_manual_text()),
            task=mas_message.task_description,
            outcome=outcome,
            key_steps=key_steps,
            fail_reason=fail_reason,
            trajectory=mas_message.get_extra_field('clean_traj') or mas_message.task_trajectory,
        )
        messages = [
            Message(
                'system',
                get_episode_reflection_system_prompt(
                    manual_context.get('profile', self.manual_state.get('profile', self._task_profile)),
                    task_success=bool(mas_message.label),
                ),
            ),
            Message('user', user_prompt),
        ]
        reflection_payload: dict[str, Any] | None = None
        last_parse_error: Optional[Exception] = None
        max_parse_attempts = 3
        for attempt in range(1, max_parse_attempts + 1):
            response = self.llm_model(
                messages=messages,
                max_tokens=self._manual_reflection_max_tokens,
            )
            try:
                reflection_payload = self._parse_json_response(response)
                break
            except Exception as e:
                last_parse_error = e
                print(
                    f"[EmbodiSkill] Failed to parse episode reflection response on "
                    f"attempt {attempt}/{max_parse_attempts}: {e}"
                )

        if reflection_payload is None:
            reflection_payload = {
                'is_empty': True,
                'reason': f'Failed to parse structured reflection after {max_parse_attempts} attempts.',
            }

        is_empty_reflection = bool(reflection_payload.get('is_empty', False))
        reflection_type, should_modify_manual, should_add_execution_note = self._derive_reflection_flags(
            reflection_type=reflection_payload.get('reflection_type', ''),
            task_success=bool(mas_message.label),
            is_empty_reflection=is_empty_reflection,
        )
        if is_empty_reflection:
            empty_reason = str(reflection_payload.get('reason', '')).strip()
            if empty_reason:
                print(f"[EmbodiSkill] Empty episode reflection discarded: {empty_reason}")
            else:
                print("[EmbodiSkill] Empty episode reflection discarded.")
            return None

        target_section = str(reflection_payload.get('target_section', '')).strip()
        target_skill = str(reflection_payload.get('target_skill', '')).strip()
        reflection_type, target_section, target_skill = self._validate_reflection_target(
            reflection_type=reflection_type,
            target_section=target_section,
            target_skill=target_skill,
        )
        reflection_type, should_modify_manual, should_add_execution_note = self._derive_reflection_flags(
            reflection_type=reflection_type,
            task_success=bool(mas_message.label),
            is_empty_reflection=is_empty_reflection,
        )
        reflection = {
            'id': self._next_reflection_id(),
            'epoch_id': self._current_epoch_id,
            'task_main': mas_message.task_main,
            'task_description': mas_message.task_description,
            'task_profile': manual_context.get('profile', self.manual_state.get('profile', self._task_profile)),
            'manual_version': int(manual_context.get('version', self.manual_state.get('version', 1))),
            'manual_text': manual_context.get('manual_text', self.get_active_manual_text()),
            'is_empty': is_empty_reflection,
            'empty_reason': str(reflection_payload.get('reason', '')).strip() if is_empty_reflection else '',
            'reflection_type': reflection_type,
            'target_section': target_section,
            'target_skill': target_skill,
            'thinking': str(reflection_payload.get('thinking', '')).strip(),
            'content': str(reflection_payload.get('content', '')).strip(),
            'should_modify_manual': should_modify_manual,
            'should_add_execution_note': should_add_execution_note,
            'trajectory': mas_message.get_extra_field('clean_traj') or mas_message.task_trajectory,
            'key_steps': key_steps,
            'fail_reason': fail_reason,
            'used_in_manual_version': None,
            'created_at_utc': self._utc_now(),
        }
        self.reflections_skill.append(reflection)
        self._save_reflections()
        return reflection

    def _serialize_pending_reflections(self, pending_reflections: list[dict[str, Any]]) -> str:
        lines = []
        for reflection in pending_reflections:
            lines.append(
                "\n".join([
                    f"- Reflection ID: {reflection.get('id')}",
                    f"  Type: {reflection.get('reflection_type')}",
                    f"  Target Section: {reflection.get('target_section')}",
                    f"  Target Skill: {reflection.get('target_skill')}",
                    f"  Content: {str(reflection.get('content', '')).strip()}",
                ])
            )
        return "\n\n".join(lines)

    def _split_revision_reflections(
        self,
        pending_reflections: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        section_types = {"S_NEW", "S_BETTER", "FAIL_SKILL"}
        execution_note_types = {"FAIL_EXECUTION"}

        section_reflections = [
            reflection for reflection in pending_reflections
            if str(reflection.get('reflection_type', '')).strip().upper() in section_types
        ]
        execution_note_reflections = [
            reflection for reflection in pending_reflections
            if str(reflection.get('reflection_type', '')).strip().upper() in execution_note_types
        ]
        return section_reflections, execution_note_reflections

    def _attempt_manual_revision_phase(
        self,
        *,
        epoch_id: Optional[int],
        phase: str,
        system_prompt: str,
        user_prompt: str,
        input_count: int,
        input_label: str,
        attempt_start: int = 1,
        max_parse_attempts: int = 3,
    ) -> tuple[Optional[dict[str, Any]], Optional[Exception]]:
        messages = [
            Message('system', system_prompt),
            Message('user', user_prompt),
        ]
        revision_payload: Optional[dict[str, Any]] = None
        last_parse_error: Optional[Exception] = None

        for attempt in range(1, max_parse_attempts + 1):
            response = self.llm_model(
                messages=messages,
                max_tokens=self._manual_revision_max_tokens,
            )

            try:
                revision_payload = self._parse_json_response(response)
                break
            except Exception as e:
                last_parse_error = e
                print(
                    f"[EmbodiSkill] Failed to parse manual revision phase `{phase}` on "
                    f"attempt {attempt}/{max_parse_attempts}: {e}."
                )

        return revision_payload, last_parse_error

    def _compress_sections_for_retry(
        self,
        *,
        epoch_id: Optional[int],
        task_profile: str,
        version: int,
        sections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        section_item_count = self._count_section_items(sections)
        section_payload = self._run_manual_revision_phase(
            epoch_id=epoch_id,
            phase="sections_retry_compaction",
            system_prompt=get_sections_refactor_system_prompt(task_profile),
            user_prompt=EmbodiSkillPrompts.sections_refactor_user_prompt.format(
                manual_text=self._format_sections_text(
                    profile=task_profile,
                    version=version,
                    sections=sections,
                ),
            ),
            input_count=section_item_count,
            input_label="section items",
            allow_skill_compaction_retry=False,
        )
        normalized_sections_state = self._normalize_manual_state({
            'profile': task_profile,
            'version': version,
            'updated_at_utc': self._utc_now(),
            'sections': section_payload.get('sections', []),
            'execution_notes': self.manual_state.get('execution_notes', []),
        })
        return normalized_sections_state.get('sections', sections)

    def _compress_execution_notes_for_retry(
        self,
        *,
        epoch_id: Optional[int],
        task_profile: str,
        version: int,
        sections: list[dict[str, Any]],
        execution_notes: list[Any],
    ) -> list[dict[str, str]]:
        execution_note_count = self._count_execution_notes(execution_notes)
        execution_notes_payload = self._run_manual_revision_phase(
            epoch_id=epoch_id,
            phase="execution_notes_retry_compaction",
            system_prompt=get_execution_notes_refactor_system_prompt(task_profile),
            user_prompt=EmbodiSkillPrompts.execution_notes_refactor_user_prompt.format(
                manual_sections_text=self._format_sections_text(
                    profile=task_profile,
                    version=version,
                    sections=sections,
                ),
                execution_notes_text=self._format_execution_notes_text(execution_notes),
            ),
            input_count=execution_note_count,
            input_label="execution notes",
            allow_skill_compaction_retry=False,
        )
        normalized_notes_state = self._normalize_manual_state({
            'profile': task_profile,
            'version': version,
            'updated_at_utc': self._utc_now(),
            'sections': sections,
            'execution_notes': execution_notes_payload.get('execution_notes', []),
        })
        return normalized_notes_state.get('execution_notes', execution_notes)

    def _run_manual_revision_phase(
        self,
        *,
        epoch_id: Optional[int],
        phase: str,
        system_prompt: str,
        user_prompt: str,
        input_count: int,
        input_label: str,
        allow_skill_compaction_retry: bool = True,
        skill_compaction_retry_builder: Optional[Callable[[], str]] = None,
    ) -> dict[str, Any]:
        max_parse_attempts = 3
        revision_payload, last_parse_error = self._attempt_manual_revision_phase(
            epoch_id=epoch_id,
            phase=phase,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_count=input_count,
            input_label=input_label,
            attempt_start=1,
            max_parse_attempts=max_parse_attempts,
        )

        if revision_payload is None and allow_skill_compaction_retry and skill_compaction_retry_builder is not None:
            print(
                f"[EmbodiSkill] Manual revision phase `{phase}` failed after {max_parse_attempts} attempts; "
                "triggering one-time skill compaction before retrying."
            )
            compacted_user_prompt = skill_compaction_retry_builder()
            revision_payload, last_parse_error = self._attempt_manual_revision_phase(
                epoch_id=epoch_id,
                phase=phase,
                system_prompt=system_prompt,
                user_prompt=compacted_user_prompt,
                input_count=input_count,
                input_label=input_label,
                attempt_start=max_parse_attempts + 1,
                max_parse_attempts=max_parse_attempts,
            )

        if revision_payload is None:
            raise RuntimeError(
                f"Manual revision phase `{phase}` JSON parsing failed after {max_parse_attempts} attempts"
                f"{' + one skill compaction retry window' if allow_skill_compaction_retry and skill_compaction_retry_builder is not None else ''} "
                f"for epoch {epoch_id}. Last error: {last_parse_error}"
            )

        return revision_payload

    def _maybe_refactor_manual_content(
        self,
        *,
        epoch_id: Optional[int],
        task_profile: str,
        version: int,
        sections: list[dict[str, Any]],
        execution_notes: list[Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
        revised_sections = copy.deepcopy(sections)
        revised_execution_notes = copy.deepcopy(execution_notes)
        triggered_phases: list[str] = []

        section_item_count = self._count_section_items(revised_sections)
        if section_item_count > self._manual_sections_refactor_threshold:
            print(
                f"[EmbodiSkill] Trigger sections refactor: {section_item_count} items > "
                f"{self._manual_sections_refactor_threshold}."
            )
            section_payload = self._run_manual_revision_phase(
                epoch_id=epoch_id,
                phase="sections_refactor",
                system_prompt=get_sections_refactor_system_prompt(task_profile),
                user_prompt=EmbodiSkillPrompts.sections_refactor_user_prompt.format(
                    manual_text=self._format_sections_text(
                        profile=task_profile,
                        version=version,
                        sections=revised_sections,
                    ),
                ),
                input_count=section_item_count,
                input_label="section items",
            )
            normalized_sections_state = self._normalize_manual_state({
                'profile': task_profile,
                'version': version,
                'updated_at_utc': self._utc_now(),
                'sections': section_payload.get('sections', []),
                'execution_notes': revised_execution_notes,
            })
            revised_sections = normalized_sections_state.get('sections', revised_sections)
            triggered_phases.append("sections_refactor")

        execution_note_count = self._count_execution_notes(revised_execution_notes)
        if execution_note_count > self._execution_notes_refactor_threshold:
            print(
                f"[EmbodiSkill] Trigger execution notes refactor: {execution_note_count} notes > "
                f"{self._execution_notes_refactor_threshold}."
            )
            execution_notes_payload = self._run_manual_revision_phase(
                epoch_id=epoch_id,
                phase="execution_notes_refactor",
                system_prompt=get_execution_notes_refactor_system_prompt(task_profile),
                user_prompt=EmbodiSkillPrompts.execution_notes_refactor_user_prompt.format(
                    manual_sections_text=self._format_sections_text(
                        profile=task_profile,
                        version=version,
                        sections=revised_sections,
                    ),
                    execution_notes_text=self._format_execution_notes_text(revised_execution_notes),
                ),
                input_count=execution_note_count,
                input_label="execution notes",
            )
            normalized_notes_state = self._normalize_manual_state({
                'profile': task_profile,
                'version': version,
                'updated_at_utc': self._utc_now(),
                'sections': revised_sections,
                'execution_notes': execution_notes_payload.get('execution_notes', []),
            })
            revised_execution_notes = normalized_notes_state.get('execution_notes', revised_execution_notes)
            triggered_phases.append("execution_notes_refactor")

        return revised_sections, revised_execution_notes, triggered_phases

    def revise_manual(self, epoch_id: Optional[int] = None, success_rate: Optional[float] = None) -> Optional[dict[str, Any]]:
        current_version = int(self.manual_state.get('version', 1))
        active_epoch_id = epoch_id if epoch_id is not None else self._current_epoch_id
        self._set_current_epoch_id(active_epoch_id)
        pending_reflections = [
            reflection for reflection in self.reflections_skill
            if reflection.get('used_in_manual_version') is None
            and not reflection.get('is_empty', False)
            and (
                active_epoch_id is None
                or self._reflection_belongs_to_epoch(reflection, active_epoch_id)
            )
        ]
        task_profile = self.manual_state.get('profile', self._task_profile)
        section_reflections, execution_note_reflections = self._split_revision_reflections(pending_reflections)

        revised_sections = copy.deepcopy(self.manual_state.get('sections', []))
        revised_execution_notes = copy.deepcopy(self.manual_state.get('execution_notes', []))
        if section_reflections:
            section_payload = self._run_manual_revision_phase(
                epoch_id=epoch_id,
                phase="sections",
                system_prompt=get_sections_revision_system_prompt(task_profile),
                user_prompt=EmbodiSkillPrompts.sections_revision_user_prompt.format(
                    manual_text=self.get_active_manual_sections_text(),
                    reflections_text=self._serialize_pending_reflections(section_reflections),
                ),
                input_count=len(section_reflections),
                input_label="reflections",
                skill_compaction_retry_builder=lambda: EmbodiSkillPrompts.sections_revision_user_prompt.format(
                    manual_text=self._format_sections_text(
                        profile=task_profile,
                        version=current_version,
                        sections=self._compress_sections_for_retry(
                            epoch_id=epoch_id,
                            task_profile=task_profile,
                            version=current_version,
                            sections=self.manual_state.get('sections', []),
                        ),
                    ),
                    reflections_text=self._serialize_pending_reflections(section_reflections),
                ),
            )
            normalized_sections_state = self._normalize_manual_state({
                'profile': task_profile,
                'version': current_version + 1,
                'updated_at_utc': self._utc_now(),
                'sections': section_payload.get('sections', []),
                'execution_notes': revised_execution_notes,
            })
            revised_sections = normalized_sections_state.get('sections', revised_sections)

        if execution_note_reflections:
            execution_notes_payload = self._run_manual_revision_phase(
                epoch_id=epoch_id,
                phase="execution_notes",
                system_prompt=get_execution_notes_revision_system_prompt(task_profile),
                user_prompt=EmbodiSkillPrompts.execution_notes_revision_user_prompt.format(
                    manual_sections_text=self._format_sections_text(
                        profile=task_profile,
                        version=current_version + 1,
                        sections=revised_sections,
                    ),
                    execution_notes_text=self.get_active_execution_notes_text(),
                    reflections_text=self._serialize_pending_reflections(execution_note_reflections),
                ),
                input_count=len(execution_note_reflections),
                input_label="reflections",
                skill_compaction_retry_builder=lambda: EmbodiSkillPrompts.execution_notes_revision_user_prompt.format(
                    manual_sections_text=self._format_sections_text(
                        profile=task_profile,
                        version=current_version + 1,
                        sections=revised_sections,
                    ),
                    execution_notes_text=self._format_execution_notes_text(
                        self._compress_execution_notes_for_retry(
                            epoch_id=epoch_id,
                            task_profile=task_profile,
                            version=current_version,
                            sections=self.manual_state.get('sections', []),
                            execution_notes=self.manual_state.get('execution_notes', []),
                        )
                    ),
                    reflections_text=self._serialize_pending_reflections(execution_note_reflections),
                ),
            )
            normalized_notes_state = self._normalize_manual_state({
                'profile': task_profile,
                'version': current_version + 1,
                'updated_at_utc': self._utc_now(),
                'sections': revised_sections,
                'execution_notes': execution_notes_payload.get('execution_notes', []),
            })
            revised_execution_notes = normalized_notes_state.get('execution_notes', revised_execution_notes)

        revised_sections, revised_execution_notes, refactor_phases = self._maybe_refactor_manual_content(
            epoch_id=epoch_id,
            task_profile=task_profile,
            version=current_version + 1,
            sections=revised_sections,
            execution_notes=revised_execution_notes,
        )

        if not pending_reflections and not refactor_phases:
            print(
                f"[EmbodiSkill] No pending reflections or refactor work for epoch {active_epoch_id}; skip manual revision."
            )
            return None

        revised_state = self._normalize_manual_state({
            'profile': task_profile,
            'version': current_version + 1,
            'updated_at_utc': self._utc_now(),
            'sections': revised_sections,
            'execution_notes': revised_execution_notes,
        })

        self.manual_state = revised_state
        self._save_manual_state()
        next_version = int(self.manual_state.get('version', current_version + 1))
        if pending_reflections:
            for reflection in pending_reflections:
                reflection['used_in_manual_version'] = next_version
            self._save_reflections()
        self.current_manual_context = None
        return self.manual_state


    def _extract_mas_message(self, mas_message: MASMessage) -> MASMessage:

        mas_message_copy: MASMessage = copy.deepcopy(mas_message)
        state_chain: StateChain = mas_message_copy.chain_of_states
        
        for state_id in reversed(range(len(state_chain))):
            if state_chain.get_state(state_id).graph.get('reward', 0) < 0:
                state_chain.pop_state(state_id)
        
        trajectory = ''
        for state in state_chain:
            # trajectory += f'> {state.graph['action']}\n{state.graph['observation']}\n'
            trajectory += f"> {state.graph['action']}\n{state.graph['observation']}\n"
        
        if mas_message_copy.label == True:
            mas_message_copy.task_trajectory = trajectory

        
        trajectory = re.sub(r'\d+', '', trajectory)
        mas_message_copy.add_extra_field('clean_traj', trajectory)


        key_steps_prompt: str = EmbodiSkillPrompts.extract_true_traj_user_prompt.format(
            task=mas_message_copy.task_description,
            trajectory=mas_message_copy.get_extra_field('clean_traj')
        )
        key_steps_messages: list[Message] = [
            Message('system', EmbodiSkillPrompts.extract_true_traj_system_prompt),
            Message('user', key_steps_prompt),
        ]

        jobs: list[tuple[str, list[Message], float]] = [('key_steps', key_steps_messages, 0.1)]
        if mas_message_copy.label == False:
            jobs.append(('fail_reason', self._build_detect_mistakes_messages(mas_message_copy), 0.0))

        responses = self._parallel_map(
            jobs,
            lambda job: (job[0], self.llm_model(job[1], temperature=job[2]))
        )
        for field_name, response in responses:
            mas_message_copy.add_extra_field(field_name, response)
        
        return mas_message_copy
    
    
    def _build_detect_mistakes_messages(self, mas_message: MASMessage) -> list[Message]:
        user_prompt: str = EmbodiSkillPrompts.detect_mistakes_user_prompt.format(
            task=mas_message.task_description,
            trajectory=mas_message.get_extra_field('clean_traj')
        )
        return [
            Message('system', EmbodiSkillPrompts.detect_mistakes_system_prompt),
            Message('user', user_prompt),
        ]

    def _detect_mistakes(self, mas_message: MASMessage) -> str:
        return self.llm_model(self._build_detect_mistakes_messages(mas_message))

    def backward(self, reward: bool):
        self.skill_cache = []
        self.skill_cache = self.skill_cache
    
    @property
    def skill_size(self):
        num_records = self.main_skill.get()["ids"]
        return len(num_records)
    
@dataclass
class TaskLayer:
    
    working_dir: str
    namespace: str
    task_storage: Chroma
    
    def __post_init__(self):
        self.similarity_threshold = 0.7

        self._graph_pic_save_path: str = os.path.join(self.working_dir, 'graph.png')
        self._node_match_save_path: str = os.path.join(self.working_dir, 'match_nodes.txt')
        self._graph_save_path: str = os.path.join(self.working_dir, f'{self.namespace}_graph.pkl')

        if os.path.exists(self._graph_save_path):
            with open(self._graph_save_path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"Graph loaded from {self._graph_save_path}")
        else:
            self.graph = nx.Graph()
            print("New empty graph created")

    def add_task_node(self, task_main: str) -> None:
        """Add a task node to the task graph.

        Args:
            task_main (str): task name
        """
        if task_main in self.graph:
            return  

        self.graph.add_node(task_main)

        results: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(
            query=task_main,
            k=10 
        )
        
        for doc, distance in results:
            similarity = 1 - distance
            if similarity < self.similarity_threshold:
                continue  

            neighbor = doc.page_content

            if neighbor not in self.graph:
                self.graph.add_node(neighbor)

            self.graph.add_edge(task_main, neighbor, weight=similarity) 
        
        self._index_done()
 
    def retrieve_related_task(self, query_task: str, node_num: int, hop: int = 1) -> list[str]:
        """
        Retrieve related tasks from the graph based on similarity and local neighborhood expansion.

        Args:
            query_task (str): The task used as the query input.
            node_num (int): The number of top similar tasks to retrieve based on similarity scores.
            hop (int, optional): The number of hops used to expand the neighborhood in the graph. Defaults to 1.

        Returns:
            list[str]: A list of related task nodes, including top similar tasks and their neighbors within the given hop.
        """
        tasks: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(query=query_task, k=node_num)
        top_nodes = [doc[0].page_content for doc in tasks]

        related_nodes = set(top_nodes)
        for node in top_nodes:
            neighbours = nx.single_source_shortest_path_length(self.graph, node, cutoff=hop).keys()
            related_nodes.update(neighbours)
        return list(related_nodes)
    
    def cluster_tasks(self) -> None:
        """
        Perform clustering on tasks in the graph using their embeddings and assign cluster IDs.

        This method extracts all nodes from the graph, computes embeddings for each node using the
        task storage's embedding function, and applies the FINCH clustering algorithm with cosine similarity.
        """
        nodes = list(self.graph.nodes)

        embeddings = []
        valid_nodes = []

        for node in nodes:
            embedding = self.task_storage._embedding_function.embed_query(node)  
            if embedding is not None:
                embeddings.append(embedding)
                valid_nodes.append(node)

        X = np.vstack(embeddings)

        try:
            # FINCH returns (cluster_labels, num_clusters, req_c)
            # We only need the finest partition (first column of cluster_labels)
            c, num_clust, req_c = FINCH(X, distance='cosine', verbose=False)
            labels = c[:, 0]  # Use the finest partition
        except Exception as e:
            print(f"FINCH clustering failed: {e}")
            labels = np.zeros(len(valid_nodes), dtype=int)

        for node, label in zip(valid_nodes, labels):
            self.graph.nodes[node]['cluster_id'] = int(label)
        self._index_done()

    def _index_done(self) -> None:
        
        with open(self._graph_save_path, "wb") as f:
            pickle.dump(self.graph, f)

    def __iter__(self) -> Iterable[tuple[str, int]]: 
        return ((node, self.graph.nodes[node]['cluster_id']) for node in self.graph.nodes)

    
