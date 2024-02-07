start=0
end=${1:-3}
for i in $(seq $start $end); do
    sbatch /scratch/bcfp/asingh15/dpo/scripts/slurm_scripts/sbatch.sh $i
done