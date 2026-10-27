import traceback

from habitat.core.agent import Agent

from src.utils.logger import Logger
from src.utils.obs_handler import ObsHandler
from src.mapping.map_handler import MapHandler
from src.llm.gpt_handler import GPTPromptHandler
from src.planning.path_handler import PathHandler
from src.utils.misc import euclidean_dist

class BaseAgent(Agent):
    def __init__(self):
        pass

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, observations):
        raise NotImplementedError
    

class ReliableAgent(BaseAgent):
    def __init__(self, config) -> None:
        self._config = config

        self._logger = Logger()
        self._obs_handler = ObsHandler(config)
        self._llm_handler = GPTPromptHandler(
            config,
            rgb_path=self._obs_handler.get_attr("rgb_path"),
            logger=self._logger,
        )
        self._map_handler = MapHandler(
            config,
            rgb_list=self._obs_handler.get_attr("rgb_list"),
            depth_list=self._obs_handler.get_attr("depth_list"),
            obs_handler=self._obs_handler,
            llm_handler=self._llm_handler,
        )
        self._path_handler = PathHandler()
        self._logger = Logger()
        
        self._rgb_list = self._obs_handler.get_attr("rgb_list")
        self._d_success = config.COMMON.SUCCESS_DISTANCE

    def reset(self, env) -> None:
        self._env = env

        self._logger.reset(
            self._config,
            env.current_episode.episode_id,
        )
        self._map_handler.reset(
            env.sim.get_agent_state(0).position,
            env.sim.get_agent_state(0).rotation,
            env.sim.pathfinder,
        )
        self._llm_handler.reset()
        self._path_handler.reset(env.sim, self._d_success)
        self._logger.reset(
            self._config,
            env.current_episode.episode_id,
        )
        self._load_action = self._config.LOG.USE_LOG_ACTION
        self._action = None

        self._subgoal_pos = None
        self._llm_max_call = self._config.LLM.MAX_SUBGOAL_CALL

        self._obs_list = []

    def act(self, observations) -> int:
        try:
            if self._load_action:
                actions = self._logger.get_logged_action()
                self._load_action = False
                if len(actions) > 0:
                    self._action = actions
            if self._action is not None:
                action = self._action.pop(0)
                if action == -1:
                    self._obs_handler.save_obs_img(observations)
                    action = self._action.pop(0)
                if action == 0:
                    self._obs_handler.save_obs_img(observations)
                return action

            pos = self._env.sim.get_agent_state(0).position
            rot = self._env.sim.get_agent_state(0).rotation
            self._obs_handler.save_obs_img(observations)

            force_update = (self._subgoal_pos is None) or (
                self._subgoal_pos is not None and 
                euclidean_dist(self._subgoal_pos, pos)[0] <= self._d_success
            )

            if force_update:
                # check if the navigation is finished
                # after every subgoal is reached
                if self._llm_max_call < 0:
                    msg = "Function act(reliable_agent.py) error: Max LLM call reached in subgoal setting, stop the navigation"
                    self._logger.log(msg, self._logger.LOG, self._logger.UNKNOWN_ERROR)
                    print(msg)
                    return 0

                if self._subgoal_pos is not None:
                    finish_flag = self._llm_handler.check_finish(
                        observations["instruction"]['text'],
                    )
                    if finish_flag > 0:
                        return 0
                    elif finish_flag == -1:
                        msg = "Function act(reliable_agent.py) error: Max LLM call reached, error in checking finish"
                        self._logger.log(msg, self._logger.LOG, self._logger.UNKNOWN_ERROR)
                        print(msg)
                        return 0

                waypoint_candidates = self._llm_handler.get_waypoint()
                if len(waypoint_candidates) == 0:
                    msg = "Function act(reliable_agent.py) error: Max LLM call reached, error in getting waypoint candidates"
                    self._logger.log(msg, self._logger.LOG, self._logger.UNKNOWN_ERROR)
                    print(msg)
                    return 0
                
                self._map_handler.update(
                    pos, rot, observations, waypoint_candidates,
                )

                imgs_path = self._obs_handler.get_img_path(waypoint_candidates)
                
                obj_list = self._map_handler.get_obj_list(
                    waypoint_candidates, pos, rot
                )
                incoming_direction = self._map_handler.get_incoming_direction(
                    pos, rot, waypoint_candidates,
                )
                
                flag = False
                for _ in range(self._llm_handler._max_single_call):
                    self._llm_handler.add_incoming_warning(incoming_direction)

                    subgoal = self._llm_handler.get_subgoal(
                        observations["instruction"]['text'],
                        waypoint_candidates,
                        imgs_path,
                        obj_list,
                    )
                    if (
                        subgoal["action"] is None or 
                        subgoal["direction"] is None
                    ):
                        msg = "Function act(reliable_agent.py) error: Max LLM call reached, error in getting subgoal"
                        self._logger.log(msg, self._logger.LOG, self._logger.UNKNOWN_ERROR)
                        print(msg)
                        return 0
                    
                    is_finish, navigable_type, subgoal_pos = self._map_handler.get_subgoal_pos(
                        subgoal, pos, rot, observations,
                    )

                    if is_finish:
                        return 0
                    
                    # for situation that the landmark or direction is not navigable,
                    # or the landmark needs double check
                    if navigable_type == 0:
                        self._llm_handler.add_unpassable_warning(
                            subgoal["landmark"], subgoal["direction"]
                        )
                        self._llm_handler.remove_trajectory()
                        continue
                    elif navigable_type == 2:
                        self._llm_handler.add_double_check_warning(
                            subgoal["action"], subgoal["direction"], subgoal["landmark"]
                        )
                        self._llm_handler.remove_trajectory()
                        continue

                    is_navigable, action = self._path_handler.get_next_action(
                        subgoal_pos,
                    )
                    if not is_navigable:
                        self._llm_handler.add_unpassable_warning(
                            subgoal["landmark"], subgoal["direction"]
                        )
                        self._llm_handler.remove_trajectory()
                        continue
                    
                    flag = True
                    break
            
                if flag:
                    self._subgoal_pos = subgoal_pos
                    self._logger.log("-----", self._logger.ACTION)
                    self._logger.log(str(action), self._logger.ACTION)
                    self._llm_max_call -= 1
                    return action
                else:
                    print("Function act(reliable_agent.py) allert: LLM call reached max times, stop the navigation")
                    return 0
        
            else:
                _, action = self._path_handler.get_next_action(
                    self._subgoal_pos # type: ignore
                )
                self._logger.log(str(action), self._logger.ACTION)
                return action
        except Exception as e:
            msg = (
                "Function act(reliable_agent.py) error: " + str(e) + " " + 
                traceback.format_exc()
            ).replace("\n", " ")
            self._logger.log(msg, self._logger.LOG, self._logger.UNKNOWN_ERROR)
            print(msg)
            return 0

    def draw_rgb(self, observations):
        from PIL import Image
        img = observations["rgb"]
        img = Image.fromarray(img)
        self._obs_list.append(img)
        img.save("rgb.png")

    def save_video(self):
        self._logger.save_video(self._obs_list)