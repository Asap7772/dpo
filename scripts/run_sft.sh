eval "$(conda shell.bash hook)"
conda activate latent_dpo

export WANDB_API_KEY=a63af51a017582ae928e7cfd56bd21d57bb45def
export WANDB_USERNAME=asap7772
export WANDB_USER_EMAIL=singh.anikait@gmail.com
entity=asap7772
wandb_project=sft-hh-02-07

models=(qwen mistral)
lrs=(1e-7 5e-7)
datasets=[hh]
loss=sft
gradient_accumulation_steps=2
batch_size=16
eval_batch_size=8

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments"
fi


for model in ${models[@]}; do
for lr in ${lrs[@]}; do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi

    exp_num=$((exp_num+1))

    echo "Running experiment ${exp_name}"
    command="python -u train.py \
        model=$model \
        datasets=$datasets \
        loss=$loss \
        exp_name=hh_sft_model${model}_lr${lr} \
        gradient_accumulation_steps=$gradient_accumulation_steps \
        batch_size=$batch_size \
        eval_batch_size=$eval_batch_size \
        trainer=FSDPTrainer \
        sample_during_eval=false \
        lr=$lr \
        wandb.entity=$entity \
        wandb.project=$wandb_project \
        "

    echo -e "$command\n"
    if [ $dry_run = false ]; then
        $command
    fi
done
done
echo "Done running all experiments: ${exp_num} experiments run."