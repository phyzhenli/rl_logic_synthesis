# Efficient Reinforcement Learning Framework for Automated Logic Synthesis Exploration


## Build

### Step0: clone repository
```
git clone --recursive https://github.com/phyzhenli/rl_logic_synthesis.git
```

### Step1: compile abc python interface
`cd abc_py` and follow the instructions.

### Step2: compile cirkit python interface
`cd cirkit_py` and follow the instructions.

### Step3: prepare rl-baselines3-zoo environment
```
git clone https://github.com/phyzhenli/rl_logic_synthesis.git
cd rl-baselines3-zoo/
git checkout v1.5.0
cp ../utils/exp_manager.py ./utils
conda create --name rl_zoo3 python=3.7
conda activate rl_zoo3
pip install -r rl-baselines3-zoo/requirements.txt
pip install dgl
cd ../
```

## Usage
```
python3 rl-baselines3-zoo/train.py \
    --env abc-v0 \
    --log-folder logs_abc_v0 \
    --gym-packages gym_eda \
    --algo ppo \
    --env-kwargs 'bench:"abc_py/a.blif"'
```
or
```
./run_bashs/run_abc_exe_opt.sh
```

### Reference:
[1] Qian, Yu, Xuegong Zhou, Hao Zhou, and Lingli Wang. "Efficient Reinforcement Learning Framework for Automated Logic Synthesis Exploration." In 2022 International Conference on Field-Programmable Technology (ICFPT), pp. 1-6. IEEE, 2022.