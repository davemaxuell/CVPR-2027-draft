"""StreamVLN's DAgger mixed control for one episode (streamvln/streamvln_dagger.py @ e48f6ff7).

The learned policy acts, except that the expert drives a whole turn with probability beta
(StreamVLN's released collection uses p = 0, so beta = 0), and the expert takes over after an
error until the next reference waypoint is reached. Errors: a policy STOP at least 3 m from the
goal, more than 12 disagreements with the expert, or disagreements above 0.8 times the expert's
actions per reference segment. Disagreements accumulate over the episode and reset only at a
takeover. A STOP before the final waypoint is never executed; the expert's action replaces it.
The episode ends with STOP once the agent is within the final waypoint's radius.

An episode is kept for training if it ends within 0.5 m of the goal and its path-length ratio
(shortest distance / max(shortest, walked)) is below 0.93 after a takeover, or below 0.85 if the
policy never needed one: StreamVLN keeps recoveries and detours, not clean successes. Kept
episodes are trained on their executed actions, with the final frame labelled STOP.

Standard library only; the simulator side supplies the expert through `label` and `reached`.
"""
import random
from dataclasses import asdict, dataclass

STOP = 0


@dataclass(frozen=True)
class DaggerRule:
    waypoint_radius_m: float = 1.8        # ShortestPathFollower goal radius for intermediate waypoints
    goal_radius_m: float = 0.25           # GOAL_RADIUS, for the final waypoint
    far_stop_m: float = 3.0               # a policy STOP at least this far from the goal is an error
    error_ratio: float = 0.8              # disagreements / (expert actions per segment) above this is an error
    max_errors: int = 12                  # disagreements above this is an error
    keep_distance_m: float = 0.5          # MIDGOAL_RADIUS: kept episodes end this close to the goal
    keep_pl_after_takeover: float = 0.93  # RELATIVE_PATH_LENGTH_THRESHOLD
    keep_pl_clean: float = 0.85           # SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD
    default_reference_actions: int = 60   # DEFAULT_EPISODE_LENGTH, when the expert length is unknown

    def as_dict(self):
        return asdict(self)


class StreamVLNController:
    """Who acts at each step. Call `decide` once per step; `waypoint` is the current target index."""

    def __init__(self, num_waypoints, reference_actions, actions_per_turn, beta=0.0, seed=0, rule=DaggerRule()):
        if num_waypoints < 2:
            raise ValueError("a reference path needs its start and at least one waypoint")
        self.rule, self.k, self.beta, self.rng = rule, actions_per_turn, beta, random.Random(seed)
        self.num_waypoints = num_waypoints
        self.per_segment = max(1, int((reference_actions or rule.default_reference_actions) / (num_waypoints - 1)))
        self.waypoint = 1           # StreamVLN's next_waypoint_id starts at 1; point 0 is the start
        self.force_expert = False
        self.from_expert = False
        self.expert_left = 0        # remaining actions of an expert-driven turn
        self.policy_left = 0        # remaining actions of the policy's current plan
        self.errors = 0
        self.needed_help = False    # StreamVLN's model_success is False after any takeover
        self.takeovers = []
        self.finished = False
        self.last_label = None

    def decide(self, t, label, reached, policy, distance_to_goal):
        """Return (action, source) for step t; action is None if the expert label is unavailable.

        label(): expert action for the current waypoint or None; reached(): whether the expert would
        STOP for the current waypoint; policy(): (policy action, actions left in its plan). Sources:
        "policy", "expert", "takeover", "veto" (expert action replacing a policy STOP), "finish".
        As in StreamVLN, the comparison with the expert happens before waypoint progress, so a
        non-STOP policy action at a reached waypoint counts as a disagreement.
        """
        if self.expert_left == 0 and self.policy_left == 0:
            self.from_expert = self.force_expert or self.rng.random() < self.beta
        expert_action = self.last_label = label()
        if self.expert_left > 0 or self.from_expert:
            action, source = expert_action, "expert"
            self.expert_left = self.expert_left - 1 if self.expert_left > 0 else self.k - 1
        else:
            action, self.policy_left = policy()
            source = "policy"
        if action != expert_action:
            self.errors += 1
        while reached():
            self.waypoint += 1
            self.force_expert, self.expert_left = False, 0
            if self.waypoint >= self.num_waypoints:
                self.finished = True
                return STOP, "finish"
        rule = self.rule
        if ((source == "policy" and action == STOP and distance_to_goal >= rule.far_stop_m)
                or self.errors / self.per_segment > rule.error_ratio or self.errors > rule.max_errors):
            self.needed_help, self.force_expert = True, True
            self.errors, self.policy_left = 0, 0
            self.takeovers.append(t)
            action, source = label(), "takeover"
        if action == STOP:  # also the expert's STOP at a waypoint it just reached: head for the next one
            action, source = label(), "veto" if source == "policy" else source
        return action, source

    def keep(self, distance_to_goal, path_length_ratio):
        rule = self.rule
        return bool(distance_to_goal < rule.keep_distance_m and (
            (self.needed_help and path_length_ratio < rule.keep_pl_after_takeover) or path_length_ratio < rule.keep_pl_clean))
