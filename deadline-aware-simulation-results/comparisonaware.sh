tot_time=2400
for seed in $(seq 1 100); do
for scheduler in \'deadline\' \'fair\'; do
python -c "import quick; quick.main(seed=$seed, filename='compa', total_time=$tot_time, no_best_effort_batches=1000, no_workers=200, scheduler=$scheduler, average_service_time=11,no_deadline_batches=int(80.0 / 60 * $tot_time), zipf_parameter=2)"
done
for scheduler in  \'aware\'; do
for log in False; do
python -c "import quick; quick.main(seed=$seed, filename='compa', total_time=$tot_time, no_best_effort_batches=1000, no_workers=200, scheduler=$scheduler, average_service_time=11,no_deadline_batches=int(80.0 / 60 * $tot_time), zipf_parameter=2, log=$log)"
done
done
done
