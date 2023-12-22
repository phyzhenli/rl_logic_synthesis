# Efficient Reinforcement Learning Framework for Automated Logic Synthesis Exploration


## How to Cite

If you use this repository, we would appreciate a citation for the following article:
```
@article{10.1145/3632174,
author = {Qian, Yu and Zhou, Xuegong and Zhou, Hao and Wang, Lingli},
title = {An Efficient Reinforcement Learning Based Framework for Exploring Logic Synthesis},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1084-4309},
url = {https://doi.org/10.1145/3632174},
doi = {10.1145/3632174},
note = {Just Accepted},
journal = {ACM Trans. Des. Autom. Electron. Syst.},
month = {nov},
keywords = {Majority-Inverter Graph, Reinforcement learning, technology mapping, And-Inverter Graph, logic optimization}
}
```


## Build

### Step1: prepare rl-baselines3-zoo
```
git clone https://github.com/DLR-RM/rl-baselines3-zoo.git; \
cd rl-baselines3-zoo/; \
git checkout v1.5.0; \
cp ../utils/exp_manager.py ./utils; \
cd ../
```

### Step2: prepare conda environment
```
conda create --name rl_zoo3 python=3.7; \
conda activate rl_zoo3; \
pip install -r rl-baselines3-zoo/requirements.txt; \
pip install dgl
```

### Step3: download and compile abc python interface
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

### Step4: download and compile cirkit python interface
```
pip install pybind11
git clone https://github.com/phyzhenli/cirkit_py.git; \
cd cirkit_py; \
make -j16; \
cd ../
```

### Step5: download and compile iMAP
```
git clone https://gitee.com/oscc-project/iMAP.git; \
cd iMAP; \
mkdir build; \
cd build; \
cmake ..; \
make -j 16; \
cd ../../
```


## Usage
#### abc-v0:
```
PYTHONPATH=.:abc_py:cirkit_py:iMAP/ai_infra/lib python3 rl-baselines3-zoo/train.py \
    --env abc-v0 \
    --log-folder logs \
    --gym-packages gym_eda \
    --algo ppo \
    --env-kwargs 'bench:"sin.aig"'
```
#### imap-exe-v0
```
PYTHONPATH=.:abc_py:cirkit_py:iMAP/ai_infra/lib python3 rl-baselines3-zoo/train.py \
    --env imap-exe-v0 \
    --log-folder logs \
    --gym-packages gym_eda \
    --algo ppo \
    --trained-agent <trained_model> \
    --env-kwargs 'imap_exe:"./iMAP/bin/imap"' 'input_file:"sin.aig"' 'step_file:"sin.aig_step"'
```
or
```
PYTHONPATH=.:abc_py:cirkit_py:iMAP/ai_infra/lib python3 rl-baselines3-zoo/train.py \
    --env imap-exe-batch-v0 \
    --log-folder logs_batch \
    --gym-packages gym_eda \
    --algo ppo \
     --env-kwargs 'imap_exe:"./iMAP/bin/imap"' 'benchs_yaml_file:"gym_eda/benchs.yaml"'
```
