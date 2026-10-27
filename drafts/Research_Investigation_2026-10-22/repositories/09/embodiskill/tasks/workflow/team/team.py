from dataclasses import dataclass

from agentkit.agents import Agent
from agentkit.skill.common import MASMessage, AgentMessage
from agentkit.orchestrator import Orchestrator
from agentkit.reasoning import ReasoningBase, ReasoningConfig
from agentkit.skill import MASSkillBase, EmbodiSkill
from agentkit.agents import Env
from agentkit.llm import LLMRequestError

from .team_prompt import TEAM_SOLVER_PROMPT 
from ..format import format_task_prompt_with_skills


@dataclass
class TeamSolver(Orchestrator):   

    def __post_init__(self):

        self.solver_name: str = 'solver'
        self.ground_truth_name: str = 'ground_truth'
        self.observers = []   

        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])

    def build_system(self, reasoning: ReasoningBase, mas_skill: MASSkillBase, env: Env, config: dict):  

        self._successful_topk: int = config.get('successful_topk', 1)
        self._failed_topk: int = config.get('failed_topk', 1)
        self._skills_topk: int = config.get('skills_topk', 3)
        self._threshold: float = config.get('threshold', 0)
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Skills Topk       : {self._skills_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")

        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_skill, MASSkillBase):
            raise TypeError("skill module must be an instance of MASSkillBase")
        
        solver_agent: Agent = Agent(
            name=self.solver_name, 
            role='solver', 
            system_instruction=TEAM_SOLVER_PROMPT.solver_system_prompt,
            reasoning_module=reasoning,
            skill_module=None
        )

        ground_truth_agent: Agent = Agent(
            name=self.ground_truth_name,
            role="ground truth agent",
            system_instruction=TEAM_SOLVER_PROMPT.ground_truth_system_prompt,
            reasoning_module=reasoning,
            skill_module=None           
        )

        env_executor = env
        
        self.hire([
            solver_agent,
            ground_truth_agent
        ])
        self.set_env(env_executor)
        self.meta_skill = mas_skill
        
    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)
    
    def schedule(
        self,
        task_config: dict,
        skills: list[str] = [],
        update_skill: bool = True,
    ) -> tuple[float, bool, dict]:
        """
        Schedules and executes a task according to the given task configuration.
        This function initializes the task context based on the configuration, retrieves relevant trajectories and skills,
        and then executes the task by interacting with the environment and agents. It also handles skill updates and feedback.
        
        Parameters:
        - task_config (dict): A dictionary containing the task configuration, including the main task and description.
        - skills (list[str]): A list of learned skills to inject into the prompt.
        - update_skill (bool): If False, run the task read-only without saving trajectories,
          extracting key steps/failure reasons, reflecting, or updating skills.
        
        Returns:
        - tuple[float, bool, dict]: Returns the final reward, whether the task was successfully completed, and trajectory data.
        """
        if task_config.get('task_main') is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required keys `task_description` in task_config")
        
        task_main: str = task_config.get('task_main')
        task_description: str = task_config.get('task_description')
        task_few_shots: list[str] = task_config.get('few_shots', [])
        # Initialize environment and agents
        env: Env = self.env
        solver: Agent = self.get_agent(self.solver_name)
        ground_truth: Agent = self.get_agent(self.ground_truth_name)
        env.reset()
        
        self.meta_skill.init_task_context(task_main, task_description) 
        manual_context = None
        if isinstance(self.meta_skill, EmbodiSkill) and self.meta_skill.uses_fixed_skill_manual():
            manual_context = self.meta_skill.get_active_manual_data()
            self.meta_skill.set_episode_manual_context(manual_context)
        
        # Retrieve successful trajectories and skills from the skill store.
        successful_trajectories: list[MASMessage]
        retrieved_skills: list[dict]

        effective_skill_topk = 0 if manual_context is not None else self._skills_topk
        successful_trajectories, _, retrieved_skills = self.meta_skill.retrieve_skill(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            skill_topk=effective_skill_topk,
            threshold=self._threshold
        )
        if manual_context is not None:
            raw_skills = manual_context.get('prompt_lines', [])
            role_skills = {agent.profile: raw_skills for agent in self.agents_team.values()}
        else:
            raw_skills = [skill for skill in retrieved_skills]
            role_skills = self._assign_skills_by_role(raw_skills)

        # Generate initial user prompt with retrieved skills
        user_prompt: str = format_task_prompt_with_skills(
            retrieved_skills=raw_skills,
            task_description=self.meta_skill.summarize(),
            skills=skills,
            few_shots=task_few_shots,
        )
        self.notify_observers(user_prompt)

        # Main loop for task execution
        action_history: list = []
        trajectory_steps: list = []  # Collect (action, observation) pairs for failure analysis 
        
        for i in range(env.max_trials):    
            
            user_prompt: str = format_task_prompt_with_skills(
                retrieved_skills=role_skills.get(solver.profile, raw_skills),
                task_description=self.meta_skill.summarize(),
                skills=skills,
                few_shots=task_few_shots,
            )
            action: str = ""
            tries = 0

            while tries < 3:
                try:
                    action = solver.response(user_prompt, self.reasoning_config)
                    if action == '':
                        if tries == 2:
                            raise RuntimeError("Solver agent returned empty action after 3 attempts.")
                        tries += 1
                        continue
                    action = env.process_action(action)
                    break
                except LLMRequestError:
                    raise
                except Exception as e:
                    print(f'Error during execution of solver agent: {e}')
                    if tries == 2:
                        raise
                    tries += 1

            name: str = solver.name
            system_instruction = solver.system_instruction
            
            if self._solver_stuck(action, action_history):
                user_prompt: str = format_task_prompt_with_skills(
                    retrieved_skills=role_skills.get(ground_truth.profile, raw_skills),
                    task_description=self.meta_skill.summarize(),
                    skills=skills,
                    few_shots=task_few_shots,
                )
                action = ""
                tries = 0
                while tries < 3:
                    try: 
                        action = ground_truth.response(user_prompt, self.reasoning_config)
                        if action == '':
                            if tries == 2:
                                raise RuntimeError("Ground-truth agent returned empty action after 3 attempts.")
                            tries += 1
                            continue
                        action = env.process_action(action)
                        break
                    except LLMRequestError:
                        raise
                    except Exception as e:
                        print(f'Error during execution of ground truth agent: {e}')
                        if tries == 2:
                            raise
                        tries += 1
                name: str = ground_truth.name
                system_instruction = ground_truth.system_instruction
            
            agent_message: AgentMessage = AgentMessage(
                agent_name=name,
                system_instruction=system_instruction,
                user_instruction=user_prompt,
                message=action,
            )
            self.meta_skill.add_agent_node(agent_message, upstream_agent_ids=[])

            observation, reward, done = env.step(action)
            action = getattr(env, 'last_matched_action', action)
            action_history.append(action)
            
            # Collect trajectory step for failure analysis
            trajectory_steps.append({
                "step": i + 1,
                "action": action,
                "observation": observation
            })
            
            step_message: str = f'Act {i + 1}: {action}\nObs {i + 1}: {observation}'
            self.notify_observers(step_message)

            self.meta_skill.move_skill_state(action, observation, reward=reward)   

            if done:  
                break

        # Final feedback and skill update.
        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        episode_reflection = None
        if update_skill:
            saved_message = self.meta_skill.save_task_context(label=final_done, feedback=final_feedback)
            if isinstance(self.meta_skill, EmbodiSkill) and self.meta_skill.uses_fixed_skill_manual():
                episode_reflection = self.meta_skill.reflect_episode(saved_message)
            self.meta_skill.backward(final_done)
        elif getattr(self.meta_skill, "current_task_context", None) is not None:
            self.meta_skill.current_task_context.label = final_done
            if final_feedback is not None:
                self.meta_skill.current_task_context.task_description += f'\n- Environment feedback\n{final_feedback}\n'
        
        # Build trajectory data for failure analysis
        trajectory_data: dict = {
            "task": task_description,
            "trajectory": trajectory_steps,
        }
        if manual_context is not None:
            trajectory_data["manual_version"] = manual_context.get("version")
        if episode_reflection is not None:
            trajectory_data["episode_reflection"] = episode_reflection

        return final_reward, final_done, trajectory_data   
    
    def _solver_stuck(self, current_action: str, action_history: list[str]) -> bool:
        """
        Determines whether the agent is stuck by repeating the same action.

        If the current action is identical to the last two actions in the history,
        the agent is considered to be stuck in a loop.

        Args:
            current_action (str): The action currently being executed by the agent.
            action_history (list[str]): A chronological list of previously taken actions.

        Returns:
            bool: True if the last two actions are the same as the current action; False otherwise.
        """
        return len(action_history) >= 2 and current_action == action_history[-1] and current_action == action_history[-2]

    def _assign_skills_by_role(self, skills: list[str]) -> dict[str, list[str]]:
        """
        Process retrieved skills to generate a dictionary matching roles to skills.

        Args:
            skills (list[str]): A list of skill strings.

        Returns:
            dict[str, list[str]]: A dictionary with roles as keys and lists of skills as values.
        """

        role_skills: dict[str, list[str]] = {}
        roles = set([agent.profile for agent in self.agents_team.values()])

        for role in roles:
            role_skills[role] = skills
        
        for role, role_specific_skills in role_skills.items():
            role_skills[role] = role_specific_skills[:self._skills_topk]
        return role_skills
