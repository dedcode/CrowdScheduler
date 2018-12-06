tot_time=2400
for seed in $(seq 5 100); do
for scheduler in \'deadline\' \'fair\'; do
python -c "import quick; quick.main(seed=$seed, filename='comp', total_time=$tot_time, no_best_effort_batches=1000, no_workers=200, scheduler=$scheduler, average_service_time=11,no_deadline_batches=int(80.0 / 60 * $tot_time), zipf_parameter=2)"
done
for scheduler in  \'saware\'; do
for log in True; do
python -c "import quick; quick.main(seed=$seed, filename='comp', total_time=$tot_time, no_best_effort_batches=1000, no_workers=200, scheduler=$scheduler, average_service_time=11,no_deadline_batches=int(80.0 / 60 * $tot_time), zipf_parameter=2, log=$log)"
done
done
done
