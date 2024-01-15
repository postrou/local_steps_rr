import argparse
import datetime

from simple_slurm import Slurm


now = datetime.datetime.now()
date = now.date()
h, m, s = now.hour, now.minute, now.second

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    dataset = args.dataset
    alpha_shift = args.alpha

    job_name = f'crr_{dataset}_a_shift_{alpha_shift}'

    output_name = f'slurm_outputs/{job_name}_{Slurm.JOB_ARRAY_MASTER_ID}_{date}_{h}:{m}:{s}.out'
    if args.gpu:
        slurm = Slurm(
            cpus_per_task=1,
            mem='10G',
            qos='gpu-8',
            partition='gpu',
            gres='gpu:1',
            job_name=job_name,
            output=output_name,
            time=datetime.timedelta(days=0, hours=3, minutes=0, seconds=0),
        )
    else:
        slurm = Slurm(
            cpus_per_task=1,
            mem='10G',
            qos='cpu-4',
            partition='cpu',
            job_name=job_name,
            output=output_name,
            time=datetime.timedelta(days=0, hours=5, minutes=0, seconds=0),
        )

    script_args = [
        dataset,
        f'--alpha {alpha_shift}'
    ]

    slurm.sbatch(f"python run_crr.py {' '.join(script_args)}")