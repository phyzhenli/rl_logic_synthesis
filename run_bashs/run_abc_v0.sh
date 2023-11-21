
 # ---------------------------------------------------
time=$(date "+%Y-%m-%d_%H:%M:%S")
echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
log_folder=logs_abc_v0
mkdir -p $log_folder/stdout 
stdout_file=$log_folder/stdout/$time.log
                            
env=abc-v0                  
cir=/home/zli/Desktop/benchmarks-master/arithmetic/sin.v
env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
env_kwargs2='optimize:"mix"'
algo=ppo                    
# vec_env='subproc'
vec_env='dummy'
n_timestep='1000'
                            
# device='4'
# CUDA_VISIBLE_DEVICES=$device 
                                python3 train.py \
                                   --log-folder $log_folder \
                                    --env $env \
                                    --env-kwargs $env_kwargs1 $env_kwargs2 \
                                    --algo $algo \
                                    --vec-env $vec_env \
                                    --n-timesteps $n_timestep \
                                    --eval-freq -1 \
                                    # >$stdout_file  2>&1 & \
 
 
#  wait
# ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/apex1.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
# # ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/C1355.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
# # ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/C6288.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
# # ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/C5315.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
# # ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/dalu.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
# # ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/k2.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
# # ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/mainpla.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
# # ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/i10.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
# # ---------------------------------------------------
# time=$(date "+%Y-%m-%d_%H:%M:%S")
# echo $time                   
                            
# cd /workspace/nodeR19/abcRL_mg/
# cd mogai6/rl-baselines3-zoo/
                            
# log_folder=/workspace/nodeR19/large_experiment/logs_benchmark/
# stdout_file=/workspace/nodeR19/large_experiment/logs_benchmark/stdout/$time.log
                            
# env=abc-v0                  
# cir=/workspace/benchmark_circuits/MCNC/Combinational/blif/C7552.blif
# env_kwargs1='bench:"'$cir'"'
# env_kwargs2='optimize:"area"'
# algo=ppo                    
# vec_env='subproc'           
# n_timestep='160000'         
                            
# device='4'                  
# CUDA_VISIBLE_DEVICES=$device python3 train.py \
#                                    --log-folder $log_folder \
#                                     --env $env \
#                                     --env-kwargs $env_kwargs1 $env_kwargs2 \
#                                     --algo $algo \
#                                     --vec-env $vec_env \
#                                     --n-timesteps $n_timestep \
#                                     --eval-freq -1 \
#                                     >$stdout_file  2>&1 & \
 
 
#  wait
