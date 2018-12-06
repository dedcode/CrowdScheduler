#!/usr/bin/env bash
#source ~/deadline/bin/activate
#seed tot_time scheduler filename
seed=$1
filename=$2
tot_time=$3
scheduler=$4
python -c "import quick; quick.main(seed=$seed, filename=$filename, total_time=$tot_time, no_best_effort_batches=1000, no_workers=200, scheduler=$scheduler, average_service_time=11,no_deadline_batches=int(80.0 / 60 * $tot_time), zipf_parameter=2, log=True)"
