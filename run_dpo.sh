betas=(0.1 0.5 0.01 0.05)
lrs=(0.0000005)
batch_size=32
eval_batch_size=4
gradient_accumulation_steps=8
chkpt='/iris/u/asap7772/trl/output_checkpoints/checkpoint-7500'

for lr in "${lrs[@]}"; do
for beta in "${betas[@]}"; do
    command="python -u train.py \
        model=pythia14 \
        datasets=[af] \
        loss=dpo \
        loss.beta=$beta \
        lr=$lr \
        exp_name=alpac_dpo_pythia14_beta${beta}_lr${lr} \
        gradient_accumulation_steps=$gradient_accumulation_steps \
        batch_size=$batch_size \
        eval_batch_size=$eval_batch_size \
        trainer=FSDPTrainer \
        sample_during_eval=false \
        model.fsdp_policy_mp=bfloat16 \
        model.archive=$chkpt \
    "
    echo $command
    eval $command
done
done