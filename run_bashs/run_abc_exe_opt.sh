
work_dir="/home/zli/Desktop/rl_logic_synthesis"

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$work_dir/boost_1_77_0/stage/lib \
PYTHONPATH=$work_dir:$work_dir/abc_py:$work_dir/cirkit_py python3 $work_dir/rl-baselines3-zoo/train.py \
   --env abc-exe-opt-v0 \
   --log-folder logs_abc-exe-opt-v0 \
   --gym-packages gym_eda \
   --algo ppo \
   --env-kwargs \
   'options_yaml_file:"gym_eda/abc_exe.yaml"'