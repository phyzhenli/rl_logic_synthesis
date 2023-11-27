from gym_eda.abc_env import AbcEnv, AbcEnvGraph
from gym_eda.abc_exe import AbcExeOpt
# from gym_eda.cirkit_env import CirkitEnv
from gym_eda.cirkit_env_v1 import CirkitEnv, CirkitEnv_step
from gym_eda.abc_env_asic import AbcEnv_ASIC, AbcEnv_ASIC_step
from gym_eda.imap_env import iMAPEnv
from gym_eda.imap_exe import iMAPExe
from gym_eda.imap_exe_batch import iMAPExeBatch


from gym.envs.registration import register

register(
    id='abc-v0',
    entry_point='gym_eda:AbcEnv',
)

register(
    id='abc-exe-opt-v0',
    entry_point='gym_eda:AbcExeOpt',
)

register(
    id='abcg-v0',
    entry_point='gym_eda:AbcEnvGraph',
)

register(
    # id='cirkit-v0',   # 原先
	id='ls-cirkit-v1', 
    entry_point='gym_eda:CirkitEnv',
)

register(
	id='ls-cirkit-step-v1', 
    entry_point='gym_eda:CirkitEnv_step',
)

register(
    id='abc-asic-v0',
    entry_point='gym_eda:AbcEnv_ASIC',
)
register(
    id='abc-asic-step-v0',
    entry_point='gym_eda:AbcEnv_ASIC_step',
)


register(
    id='imap-v0',
    entry_point='gym_eda:iMAPEnv',
)

register(
    id='imap-exe-v0',
    entry_point='gym_eda:iMAPExe',
)