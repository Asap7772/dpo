start=2955300
end=${1:-2955307}
for i in $(seq $start $end); do
    scancel $i
done