import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

class PathHandler():
    def __init__(self):
        pass

    def reset(self, sim, d_success: float):
        self._planner = ShortestPathFollower(
            sim, d_success, return_one_hot=True, stop_on_error=False
        )
        self._waypoint = None

    def get_next_action(
            self,
            subgoal: np.ndarray,
    ) -> tuple[bool, int]:
        try:
            action = self._planner.get_next_action(subgoal)
            return True, np.argmax(action) # type: ignore
        except Exception as e:
            print("Function get_next_action(path_handler.py) allert: The subgoal pos is not navigable")
            return False, -1
        
        


    
    