#!/bin/bash

export PYTHONUNBUFFERED=1

temp=(0)
job=0

for param in "${temp[@]}"; do
    for seed in 1 2 3 4 5 6 7 8; do
        name=walker-$seed-thompson-fixed
        project_name=baseline-pomp2
        save_dir=~/sfu/res/saved_models/RL/${project_name}
        mkdir -p ${save_dir}/${name}
        save_log_file=${save_dir}/${name}/out.log
            python main_maac2.py --exploration_init --cuda --save_result --save_model \
                --automatic_entropy_tuning True --see_freq 1000 \
                --env-name Walker2d-v2 --num_steps 300000 \
                --start_steps 5000 --save_model_interval 1000 \
                --model_type Naive --weight_grad 10 \
                --batch_size_pmp 256 --lr 3e-4 \
                --update_policy_times 10 --updates_per_step 10 \
                --rollout_max_length 1 --max_train_repeat_per_step 10 --min_pool_size 5000 \
                --near_n 5 --seed $seed --H 4 \
                --save_prefix $name --policy_direct_bp \
                --epsilon 0.1 \
                --n_critic 5 \
                --policy_ga_end_increase_epoch 50 --policy_ga_num_iters 10 \
                --ddp_max_delta 20 --ddp_clipk 0.025 --ddp_delta_decay_legacy --project_name $project_name --wandb_name $name > $save_log_file 2>&1 &
            job=$((job+1))
            pids[$job]=$!
            names[$job]=$name
            sleep 5
    done
done

echo ${pids[*]}
echo ${names[*]}
for pid in ${pids[*]}; do
    wait $pid
done

date
