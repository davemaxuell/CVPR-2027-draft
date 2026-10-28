"""Actual-observation lifecycle and logged base-policy fallback, without Habitat."""
import torch
if __package__:
    from .scorer import START, STOP, choose_action, validate_margin
else:
    from scorer import START, STOP, choose_action, validate_margin

_CONFIG_MARGIN = object()

class ScorerRuntime:
    """Single-episode controller. The policy adapter supplies REAL prefill features.
    commit() records the action actually executed, never an unexecuted action chunk.
    """
    def __init__(self, scorer, *, margin=_CONFIG_MARGIN):
        self.scorer = scorer.eval()
        if margin is _CONFIG_MARGIN:
            margin = scorer.settings['selection']['override_margin'] if scorer.settings else None
        validate_margin(margin)
        self.margin = margin
        self.reset()

    def reset(self):
        self.context = None
        self.step = -1
        self.committed = None
        self.terminated = False
        self.events = []

    @torch.no_grad()
    def observe(self, step, **features):
        if self.terminated or step != self.step + 1:
            raise ValueError('Reset terminated episodes; observe each real step exactly once')
        previous = START if step == 0 else self.committed
        if previous is None:
            raise ValueError('Commit the actually executed action before the next observation')
        if features['previous_action'].shape != (1,) or features['previous_action'].item() != previous:
            raise ValueError('Previous action must match the real execution record')
        memory = None if self.context is None else self.context.memory
        context = self.scorer.observe(**features, memory=memory)
        self.context, self.step, self.committed = context.detached(), step, None

    @torch.no_grad()
    def decide(self, base_action, predict_movements):
        if self.context is None or self.terminated or self.committed is not None:
            raise ValueError('Decision requires an uncommitted real observation')
        if type(base_action) is not int or not 0 <= base_action <= STOP:
            raise ValueError('Invalid base action')
        device = self.context.tokens.device
        actions = torch.arange(4, device=device)[None]
        base_index = torch.tensor([base_action], device=device)
        utility = None
        error = None
        if self.margin is not None:
            try:
                movement = predict_movements(actions[:, :3])
                expected = (1,3,self.scorer.patch_count,self.scorer.visual.feature_width)
                if not isinstance(movement, torch.Tensor) or not movement.is_floating_point() or movement.shape != expected:
                    raise ValueError(f'Expected movement predictions {expected}')
                future = torch.cat((movement.to(device), self.context.current_patches[:,None].to(movement.dtype)), dim=1)
                utility = self.scorer.score(self.context, future, actions)['utility']
            except (ValueError, TypeError, RuntimeError) as exc:
                error = f'{type(exc).__name__}: {exc}'
        selection = choose_action(utility, actions, base_index, self.margin)
        self.events.append(dict(step=self.step, reason=selection.reasons[0], prediction_error=error,
                                base_action=base_action, selected_action=int(selection.actions.item())))
        return int(selection.actions.item())

    def commit(self, executed_action):
        if self.context is None or self.committed is not None or self.terminated:
            raise ValueError('Only one real action may be committed per observation')
        if type(executed_action) is not int or not 0 <= executed_action <= STOP:
            raise ValueError('Invalid executed action')
        self.committed = executed_action
        self.terminated = executed_action == STOP
