#!/usr/bin/env bash
# ITES Safety HRAC on POLAMP — cross_dataset_test_level_1
if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

cd ../..
python main.py --domain_name Polamp \
               --env_name SafePolamp \
               --polamp_dataset cross_dataset_test_level_1 \
               --seed $seed \
               --eval_freq 30000 \
               --polamp_plot_trajectory \
               --visulazied_episode 0 \
               --man_rew_scale 1.0 \
               --goal_loss_coeff 20.0 \
               --manager_propose_freq 10 \
               --train_manager_freq 5 \
               --world_model \
               --wm_pretrain_epoches 100 \
               --wm_n_initial_exploration_steps 30000 \
               --cost_model \
               --cm_frame_stack_num 1 \
               --cost_model_batch_size 512 \
               --manager_algo "td3_adj_safe_cls" \
               --coef_safety_modelfree 10.0 \
               --a_net_discretization_koef 10.0 \
               --controller_algo "td3_img_safe" \
               --img_horizon 10 \
               --controller_safety_coef 0.001 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb \
               --cm_pretrain \
               --wm_pretrain
