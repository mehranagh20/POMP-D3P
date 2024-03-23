#!/bin/bash

job=0

# setting: eps0.5-ga_lr0.002-iter10-n_critic5-end_iter50-eps_decay_end100000-noisy5-eff1



eps=0.5
iter=10
n_critic=5
end_iter=50
ga_lr=0.002
eps_decay_end=100000
noisy_coef=5
efficient=1

# for seed in 7 5 3 1 ; do
for seed in 1; do
    name=eps${eps}-ga_lr${ga_lr}-iter${iter}-n_critic${n_critic}-end_iter${end_iter}-eps_decay_end${eps_decay_end}-noisy${noisy_coef}-eff${efficient}-seed${seed}
    setting=eps${eps}-ga_lr${ga_lr}-iter${iter}-n_critic${n_critic}-end_iter${end_iter}-eps_decay_end${eps_decay_end}-noisy${noisy_coef}-eff${efficient}-3-ws
    project_name=walker-final
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
                --noisy_critic_efficient $efficient \
                --n_critic $n_critic \
                --save_dir ${save_dir}/${name} \
                --policy_ga_lr $ga_lr \
                --epsilon_decay_end $eps_decay_end \
                --wandb_mode offline \
                --policy_ga_end_increase_epoch $end_iter --policy_ga_num_iters $iter \
                --ddp_max_delta 10 --ddp_clipk 0.025 --ddp_delta_decay_legacy --project_name $project_name --wandb_name $name
        job=$((job+1))
        pids[$job]=$!
        names[$job]=$name
        sleep 1
done
echo ${pids[*]}
echo ${names[*]}
for pid in ${pids[*]}; do
    wait $pid
done

date
