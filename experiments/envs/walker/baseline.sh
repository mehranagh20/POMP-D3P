#!/bin/bash

job=0

# eps0.3-ga_lr0.01-iter100-n_critic5-end_iter50-eps_decay_end100000_noisy10.0

eps=0.0
iter=100
n_critic=5
end_iter=50
ga_lr=0.01
eps_decay_end=100000
noisy_coef=10.0
efficent=1

for seed in 1 2 3 4 5 6 7 8; do
    name=eps${eps}-ga_lr${ga_lr}-iter${iter}-n_critic${n_critic}-end_iter${end_iter}-eps_decay_end${eps_decay_end}-noisy${noisy_coef}-seed${seed}
    setting=baseline
    project_name=walker-best
    save_dir=/home/mehran/sfu/projects/rl/POMP-D3P/results/$project_name
    mkdir -p ${save_dir}/${name}
    save_log_file=${save_dir}/${name}/out.log
    python main_maac2.py --exploration_init --cuda --save_result --save_model \
                --automatic_entropy_tuning True --see_freq 1000 \
                --env-name Walker2d-v2 --num_steps 250000 \
                --start_steps 5000 --save_model_interval 1000 \
                --model_type Naive --weight_grad 10 \
                --batch_size_pmp 256 --lr 3e-4 \
                --update_policy_times 10 --updates_per_step 10 \
                --rollout_max_length 1 --max_train_repeat_per_step 10 --min_pool_size 5000 \
                --near_n 5 --seed $seed --H 4 \
                --save_prefix $name --policy_direct_bp \
                --setting $setting \
                --epsilon $eps \
                --noisy_coef $noisy_coef \
                --noisy_critic_efficient $efficent \
                --n_critic $n_critic \
                --policy_ga_lr $ga_lr \
                --epsilon_decay_end $eps_decay_end \
                --policy_ga_end_increase_epoch $end_iter --policy_ga_num_iters $iter \
                --ddp_max_delta 10 --ddp_clipk 0.025 --ddp_delta_decay_legacy --project_name $project_name --wandb_name $name > $save_log_file 2>&1 &
        job=$((job+1))
        pids[$job]=$!
        names[$job]=$name
        sleep 5
done
echo ${pids[*]}
echo ${names[*]}
for pid in ${pids[*]}; do
    wait $pid
done

date