
cd /home/yuqian/Documents/2021_07_14_A/synthesis_workspace/nodeR19/files2/trash/rl-baselines3-zoo_05-04
cir_path='bench:"/home/yuqian/Documents/2021_07_14_A/synthesis_workspace/benchmark_circuits/FlowTune_blifs/bfly.abc.blif"'
echo $cir_path
scl_lib='cell_lib:"/home/yuqian/Documents/2021_07_14_A/synthesis_workspace/benchmark_circuits/asap7_mail.lib"'
log_folder=logs_scl1
time=$(date "+%Y-%m-%d_%H:%M:%S")
echo $time 
mkdir -p $log_folder/stdout/
stdout_file=$log_folder/stdout/$time.log
htop_id=h_$time

# * 换ppo.yml中参数
CUDA_VISIBLE_DEVICES='2' python3 train.py -f $log_folder --env abc-asic-v0 --env-kwargs $cir_path \
        'mapping:"SCL;2000"' \
        $scl_lib \
        'baseline:"dc2;dchif;dc2;dchif;dc2;dchif;dc2;dchif;dc2;dchif;       strash;ifraig;scorr;dc2;strash;dch -f;"'  \
        'max_seq_len:39'  \
        'actions:"rs 8;dc2;mapD_mst;&ifK3a_ifK3g;-z;+l"' \
        'seq_end:"time"' \
        'map_tail:"flowtune"' 'tune_actions:"None"' \
        'alias_flag:"alias_mapD_mst"' \
        'optimize:"mix"' \
        --algo ppo \
        --htop_id $htop_id \
        --vec-env 'subproc' \
        >$stdout_file  2>&1  & \



