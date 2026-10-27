"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""

import re
import math
import json
import numpy as np
from .agent_base import BaseAgent
from models.base_mllm import BaseMLLM
from .prompt import PROMPT_COT_REFLECT, SYSTEM_COT_REFLECT
from models import get_models
from typing import List, Dict, Tuple, Optional

from utils.data import AgentAction, AgentFinish, OutputParserException

FINAL_ANSWER_ACTION = "Stop"

MISSING_ANSWER_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' in LLM output"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)
VIEWPOINT_NOT_IN_NAVIGABLE_ERROR_MESSAGE = (
    "Invalid Choice: Viewpoint not in navigable viewpoints"
)
ANSWER_NOT_VALID_ERROR_MESSAGE = (
    "Invalid Choice: Answer is not a valid number"
)

class BaselineCoTReflectionAgent(BaseAgent):
    def __init__(self, env_name, env, config, logger):
        super().__init__(env_name, env, config)
        
        self.logger = logger

        model_class = get_models(config.experiment.model, logger)
        self.model: BaseMLLM = model_class()

        self.history: List[str] = None

    def parse(self, text: str, id_viewpoint: dict, action_options: dict):
        final_answer = FINAL_ANSWER_ACTION in text

        # Two-step approach for robust action parsing
        def extract_action_value(text):
            """
            Ultra-robust action parser that handles noise characters.
            Looks for: Action + some chars + : + some chars + (number or direction)
            """
            
            # Method 1: Find Action...colon pattern with flexible noise handling
            # Allows up to 10 characters of "noise" between Action and colon
            action_pattern = r"Action.{0,10}?:"
            match = re.search(action_pattern, text, re.IGNORECASE | re.DOTALL)
            
            if match:
                # Get everything after the matched Action....: part
                after_action = text[match.end():]
                
                # Look for number or direction in the next 50 characters
                # This handles cases like: Action": 5. "Bathroom... or Action***: Left some text
                search_window = after_action[:50]  # Limit search to avoid false positives
                
                # First try to find a number (prioritize numbers)
                number_match = re.search(r"(-?\d+(?:\.\d+)?)", search_window)
                if number_match:
                    return number_match.group(1)
                
                # If no number, look for direction
                direction_match = re.search(r"(Left|Right|Front|Back)", search_window, re.IGNORECASE)
                if direction_match:
                    return direction_match.group(1)
            
            # Method 2: Fallback - look for Action and number/direction anywhere in proximity
            # This is even more flexible but might have more false positives
            action_pos = text.lower().find('action')
            if action_pos != -1:
                # Look in the next 100 characters after "action"
                search_area = text[action_pos:action_pos + 100]
                
                # Find colon first, then look for number/direction after it
                colon_pos = search_area.find(':')
                if colon_pos != -1:
                    after_colon = search_area[colon_pos + 1:]
                    
                    # First try to find a number
                    number_match = re.search(r"(-?\d+(?:\.\d+)?)", after_colon)
                    if number_match:
                        return number_match.group(1)
                    
                    # If no number, look for direction
                    direction_match = re.search(r"(Left|Right|Front|Back)", after_colon, re.IGNORECASE)
                    if direction_match:
                        return direction_match.group(1)
            
            return None

        # Robust reflection and decision extraction
        def extract_robust_reflection_and_decision(text):
            reflection = None
            decision = None
            
            # Pattern for Reflection (handles <>, **, or no formatting)
            reflection_pattern = r"<?(?:\*{0,2})?Reflection(?:\*{0,2})?>?:\s*(.*?)(?=<?(?:\*{0,2})?(?:Decision|Final Decision|Action)(?:\*{0,2})?>?:|$)"
            reflection_match = re.search(reflection_pattern, text, re.IGNORECASE | re.DOTALL)
            if reflection_match:
                reflection = reflection_match.group(1).strip()
                # Clean up any remaining markdown formatting
                reflection = re.sub(r'\*{1,2}', '', reflection).strip()
            
            # Pattern for Decision/Final Decision (handles <>, **, or no formatting)
            decision_pattern = r"<?(?:\*{0,2})?(?:Final\s+)?Decision(?:\*{0,2})?>?:\s*(.*?)(?=<?(?:\*{0,2})?Action(?:\*{0,2})?>?:|$)"
            decision_match = re.search(decision_pattern, text, re.IGNORECASE | re.DOTALL)
            if decision_match:
                decision = decision_match.group(1).strip()
                # Clean up any remaining markdown formatting
                decision = re.sub(r'\*{1,2}', '', decision).strip()
            
            return reflection, decision

        # Extract action using the robust method
        action_str = extract_action_value(text)
        
        if action_str:
            action = "make_action"
            
            # Extract optional reflection and decision using robust method
            reflection, decision = extract_robust_reflection_and_decision(text)
            
            # Validate action
            if action_str.isdigit():
                if action_str not in id_viewpoint or id_viewpoint[action_str] not in self.env.env.sims[0].navigable.keys():
                    raise OutputParserException(
                        f"Could not make corresponding action: `{text}`",
                        observation=VIEWPOINT_NOT_IN_NAVIGABLE_ERROR_MESSAGE,
                        llm_output=text,
                        send_to_llm=True,
                    )
                
                return AgentAction(action, action_str, text, 
                                extra_data={'reflection': reflection, 'decision': decision})
                
            elif action_str in action_options:
                direction_dict = action_options[action_str]
        
                if isinstance(direction_dict, dict) and direction_dict:
                    action_number = next(iter(direction_dict.keys()))
                    tool_input = id_viewpoint[action_number]

                    if tool_input not in self.env.env.sims[0].navigable.keys():
                        raise OutputParserException(
                            f"Could not make corresponding action: `{text}`",
                            observation=VIEWPOINT_NOT_IN_NAVIGABLE_ERROR_MESSAGE,
                            llm_output=text,
                            send_to_llm=True,
                        )

                    return AgentAction(action, action_number, text,
                                    extra_data={'reflection': reflection, 'decision': decision})
                else:
                    raise OutputParserException(
                        f"Could not make corresponding action: `{text}`",
                        observation=VIEWPOINT_NOT_IN_NAVIGABLE_ERROR_MESSAGE,
                        llm_output=text,
                        send_to_llm=True,
                    )
                    
        elif final_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        # Only fail if no Action found
        if not re.search(r"<?(?:\*{0,2})?Action(?:\*{0,2})?>?:", text, re.IGNORECASE):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ANSWER_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        

    def init_trajectory(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        # Record the navigation path
        self.traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'llm_outputs': [],
            'details': [],
            'initial_heading': round(math.degrees(ob['heading']), 2),
            'scan': ob['scan']
        } for ob in obs]
        # Record the history of actions taken
        self.history = ["Navigation starts."]

        print(f"\nExecuting instruction:\n{obs[0]['instr_id']}: {obs[0]['instruction']}")


    def make_equiv_action(self, actions: List[dict], step) -> str:
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle
        
        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"

        # Get current agent facing angle
        cur_obs = self.env._get_obs()[0]
        cur_heading = np.rad2deg(cur_obs['heading'])

        # Get the target caption
        target_caption = cur_obs['caption'][actions[0]]

        # Extract viewpoint ID
        tool_input = cur_obs['id_viewpoint'][actions[0]]
        target = [tool_input.strip(" ").strip('"').strip("'")]

        # Get the distance from current viewpoint to target
        distance = cur_obs['candidate'][target[0]]['distance']

        # Make the action
        new_obs = self.env.step(target)[0]
        new_heading = np.rad2deg(new_obs['heading'])

        # Record the trajectory
        self.traj[0]['path'].append([target[0]])

        # Calculate the turned angle
        turned_angle = new_heading - cur_heading

        # Generate action description
        cur_heading = angle_to_left_right(normalize_angle(cur_heading))
        new_heading = angle_to_left_right(normalize_angle(new_heading))

        action_description = f'Step {step}: Turning heading direction {turned_angle:.2f} degrees from {cur_heading} to {new_heading}, and forward {distance:.2f} meters towards {target_caption}'

        return action_description, new_obs, turned_angle

    def make_action(self, action: AgentAction, step) -> str:
        """Make single action in Simulator"""
        if action.tool == 'make_action':
            raw_output = action.log
            
            try:
                # Use the extracted reflection and decision from the parse function
                if hasattr(action, 'extra_data') and action.extra_data:
                    reflection = action.extra_data.get('reflection', '')
                    decision = action.extra_data.get('decision', 'Bad')
                else:
                    # Fallback: extract reflection and decision if not available in extra_data
                    reflection, decision = self._extract_reflection_and_decision(raw_output)
                    
            except ValueError as e:
                # Handle case where reflection/decision extraction fails
                reflection = "Error extracting reflection"
                decision = "Bad"
                print(f"Warning: {str(e)}")

            if decision and "good" in decision.lower():
                # Make action
                action_description,new_obs, turned_angle = self.make_equiv_action([action.tool_input], step)

                history = f"Step {step}: {action_description} <Reflection>: {reflection} <Final Decision>: Good."
            else:
                turned_angle = None
                new_obs = self.env._get_obs()[0]

                history = "Based on the reflection, the proposed action is bad, no action will be taken."
        else:
            # For non-valid actions
            history = action.tool_input # Error observation message
            turned_angle = None
            new_obs = self.env._get_obs()[0]

        self.history.append(history)

        detail = {
            "viewpointID": action.tool_input,
            "turned_angle": turned_angle,
            "summary": new_obs["summary"], # TODO, check this if needed
            "history": self.history[-1],
        }
        self.traj[0]['details'].append(detail)

        return [new_obs]
    
    def format_input(self, input):
        prompt = PROMPT_COT_REFLECT.format(
            system=SYSTEM_COT_REFLECT,
            instruction=input["instruction"],
            history=input["history"],
            heading=round(input["heading"], 2),
            elevation=round(input["elevation"], 2),
            caption=input['caption'],
            action_options=input["action_options"],
        )

        return prompt

    def rollout(self, reset=True):
        if reset: # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajectory(obs)

        # Load the instruction
        instructions = [ob['instruction'] for ob in obs]

        self.traj[0]['instruction'] = instructions[0]

        # Rollout
        step = 0
        visited = {}
        while step < self.config.experiment.max_step:
            for i, ob in enumerate(obs):
                instruction = instructions[i]
                image_obs = ob['img_obs']['egocentric'] # Single egocentric plot
                caption = ob['caption']

                heading = np.rad2deg(ob['heading'])
                elevation = np.rad2deg(ob['elevation'])

            cur_vp = ob['viewpoint']
            if cur_vp in visited:
                visited[cur_vp] += 1
            else:
                visited[cur_vp] = 1

            if any(value > 3 for value in visited.values()):
                output = "Action: Stop. Have reached the destination and stop here."
            else:
                input = {
                    "instruction": instruction,
                    "action_options": json.dumps(ob['action_options'], indent=2),
                    "caption": caption,
                    "heading": heading,
                    "elevation": elevation,
                    "history": self.history,
                    'image': image_obs
                }

                prompt = self.format_input(input)

                # Model Inference
                output = self.model.__call__(prompt=prompt, input=input)
            self.traj[0]['llm_outputs'].append(output)

            # Parse LLMs output
            try:
                action = self.parse(output, ob['id_viewpoint'], ob['action_options'])
            except OutputParserException as e:
                observation = str(e.observation)
                text = str(e.llm_output)

                action = AgentAction("_Exception", observation, text, {})

            # If the tool chosen is the finishing tool, then we end and return.
            if isinstance(action, AgentFinish):
                self.traj[0]['history'] = self.history
                self.logger.info(f"\nStep {step}:\nLLM output: {action.log}\nAction: Finished!")
                break

            if isinstance(action, AgentAction):
                # Make action
                obs = self.make_action(action, step+1)
                print(f"\nStep {step}:\nLLM output: {action.log}\nAction: {self.history[-1]}")

            # Update history
            self.traj[0]['history'] = self.history
            step += 1

        return self.traj
