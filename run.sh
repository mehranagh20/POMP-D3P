#!/bin/bash

 python main_maac2.py --exploration_init --cuda --save_result --save_model \
                --automatic_entropy_tuning True --see_freq 1000 \
                --env-name Walker2d-v2 --num_steps 105000 \
                --ddp_num_samples $num_ddp \
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
                --n_critic $n_critic --wandb_mode offline  \
                --policy_ga_lr $ga_lr \
                --epsilon_decay_end $eps_decay_end \
                --policy_ga_end_increase_epoch $end_iter --policy_ga_num_iters $iter \
                --ddp_max_delta 10 --ddp_clipk 0.025 --ddp_delta_decay_legacy 
