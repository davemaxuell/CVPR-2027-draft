from hydra.core.config_store import ConfigStore

from mad_td.cfgs.mad_td_config import MadTdHyperparams
from mad_td.cfgs.train_config import TrainHyperparams
from mad_td.cfgs.data_config import EnvConfig

cs = ConfigStore.instance()
cs.store(group="train", name="base_train", node=TrainHyperparams)
cs.store(group="env", name="base_env", node=EnvConfig)
cs.store(group="algo", name="base_mad_td", node=MadTdHyperparams)
cs.store(group="algo", name="corrected_itervaml", node=MadTdHyperparams)
cs.store(group="algo", name="uncorrected_itervaml", node=MadTdHyperparams)
cs.store(group="algo", name="uncorrected_muzero", node=MadTdHyperparams)
cs.store(group="algo", name="corrected_muzero", node=MadTdHyperparams)
cs.store(group="algo", name="model_free", node=MadTdHyperparams)
