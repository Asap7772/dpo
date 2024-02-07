eval "$(conda shell.bash hook)"
conda activate latent_dpo

export WANDB_API_KEY=a63af51a017582ae928e7cfd56bd21d57bb45def
export WANDB_USERNAME=asap7772
export WANDB_USER_EMAIL=singh.anikait@gmail.com
entity=asap7772
wandb_project=latent-dpo-02-07-fixedid

models=(qwen mistral)
dpo_betas=(0.1 0.05)
kl_weights=(0.01 0.001)

datasets=[hh]
loss=dpo
n_samples_per_scorer=6
gradient_accumulation_steps=4
batch_size=24
eval_batch_size=18
n_eval_examples=252
dtype=bfloat16

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments"
fi


for model in ${models[@]}; do
for dpo_beta in ${dpo_betas[@]}; do
for kl_weight in ${kl_weights[@]}; do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi

    exp_num=$((exp_num+1))
    exp_name=rerun_dpovae_${model}_beta_${dpo_beta}_kl_weight_${kl_weight}_exp_${exp_num}

    echo "Running experiment ${exp_name}"
    command="python train.py \
        model=${model} \
        datasets=${datasets} \
        loss=${loss} \
        loss.beta=${dpo_beta} \
        exp_name=${exp_name} \
        n_samples_per_scorer=${n_samples_per_scorer} \
        gradient_accumulation_steps=${gradient_accumulation_steps} \
        batch_size=${batch_size} \
        eval_batch_size=${eval_batch_size} \
        n_eval_examples=${n_eval_examples} \
        trainer=BasicTrainer \
        sample_during_eval=false \
        model.policy_dtype=${dtype} \
        model.reference_dtype=${dtype} \
        activation_checkpointing=True \
        loss.kl_weight=${kl_weight} \
        wandb.entity=$entity \
        wandb.project=$wandb_project \
    "
    echo -e "$command\n"
    if [ $dry_run = false ]; then
        $command
    fi
done
done
done
echo "Done running all experiments: ${exp_num} experiments run."