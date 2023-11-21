# Efficient Reinforcement Learning Framework for Automated Logic Synthesis Exploration


## Build


### Step1: download and compile abc python interface
```
git clone https://github.com/phyzhenli/abc_py.git; \
cd abc_py; \
git clone https://github.com/berkeley-abc/abc.git; \
cd abc; \
make -j16 ABC_USE_NO_READLINE=1 ABC_USE_STDINT_H=1 ABC_USE_PIC=1 libabc.a; \
cd ../; \
make; \
cd ../
```

### Step2: download and compile cirkit python interface
```
git clone https://github.com/phyzhenli/cirkit_py.git; \
cd cirkit_py; \
make -j16; \
cd ../
```

### Step3: prepare rl-baselines3-zoo
```
git clone https://github.com/DLR-RM/rl-baselines3-zoo.git; \
cd rl-baselines3-zoo/; \
git checkout v1.5.0; \
cp ../utils/exp_manager.py ./utils; \
cd ../
```

### Step4: prepare conda environment
```
conda create --name rl_zoo3 python=3.7; \
conda activate rl_zoo3; \
pip install -r rl-baselines3-zoo/requirements.txt; \
pip install dgl
```

## Usage
```
PYTHONPATH=.:abc_py:cirkit_py python3 rl-baselines3-zoo/train.py
    --env abc-v0
    --log-folder logs
    --gym-packages gym_eda
    --algo ppo
    --env-kwargs 'bench:"abc_py/s838.blif"'
```

## Reference:
[1] Qian, Yu, Xuegong Zhou, Hao Zhou, and Lingli Wang. "Efficient Reinforcement Learning Framework for Automated Logic Synthesis Exploration." In 2022 International Conference on Field-Programmable Technology (ICFPT), pp. 1-6. IEEE, 2022.
