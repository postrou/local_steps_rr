import argparse
import datetime

from simple_slurm import Slurm
import numpy as np


now = datetime.datetime.now()
date = now.date()
h, m, s = now.hour, now.minute, now.second

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('alg', type=str)
    parser.add_argument('--n_epochs', type=int, default=250)
    parser.add_argument('--x_opt', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--cl_min', type=int, default=None, help='min clip level in log scale')
    parser.add_argument('--cl_max', type=int, default=None, help='max clip level in log scale')
    parser.add_argument('--a_min', type=int, default=None, help='min alpha in log scale')
    parser.add_argument('--a_max', type=int, default=None, help='max alpha in log scale')
    args = parser.parse_args()

    dataset = args.dataset
    alg = args.alg
    n_epochs = args.n_epochs
    is_x_opt = args.x_opt
    cl_min = args.cl_min
    cl_max = args.cl_max
    a_min = args.a_min
    a_max = args.a_max
    # clip_level_list = None
    # alpha_shift_list = None

    if alg.startswith('crr'):
        assert args.cl_min is not None and args.cl_max is not None, \
            f'You did not provide --cl_min or --cl_max for algorithm {alg}'
        # clip_level_list = np.logspace(-3, 3, 7)
        if alg == 'crr_shift':
            assert args.a_min is not None and args.a_max is not None, \
                f'You did not provide --a_min or --a_max for algorithm {alg}'
            # alpha_shift_list = np.logspace(-4, 2, 7)

    if alg.startswith('crr_shift'):
        if alg == 'crr_shift':
            job_name = f'c{cl_min}_{cl_max}_a_{a_min}_{a_max}_rr_{dataset}_{n_epochs}'
        elif alg == 'crr_shift_2':
            job_name = f'c{cl_min}_{cl_max}_shift_rr_2_{dataset}_{n_epochs}'
        elif alg == 'crr_shift_3':
            job_name = f'c{cl_min}_{cl_max}_shift_rr_3_{dataset}_{n_epochs}'
        elif alg == 'crr_shift_optf':
            job_name = f'c{cl_min}_{cl_max}_shift_rr_opt_full_{dataset}_{n_epochs}'
        elif alg == 'crr_shift_mean':
            job_name = f'c{cl_min}_{cl_max}_a_{a_min}_{a_max}_mean_rr_{dataset}_{n_epochs}'
        elif alg == 'crr_shift_saga':
            job_name = f'c{cl_min}_{cl_max}_a_{a_min}_{a_max}_saga_rr_{dataset}_{n_epochs}'
        else:
            raise NotImplementedError()
        output_name = f'slurm_outputs/{job_name}_{Slurm.JOB_ARRAY_MASTER_ID}_{date}_{h}:{m}:{s}.out'
        if args.gpu:
            slurm = Slurm(
                cpus_per_task=1,
                mem='5G',
                qos='gpu-8',
                partition='gpu',
                gres='gpu:1',
                job_name=job_name,
                output=output_name,
                time=datetime.timedelta(days=0, hours=3, minutes=0, seconds=0),
            )
        else:
            slurm = Slurm(
                cpus_per_task=50,
                mem='15G',
                qos='cpu-512',
                partition='cpu',
                job_name=job_name,
                output=output_name,
                time=datetime.timedelta(days=0, hours=12, minutes=0, seconds=0),
            )
        # slurm.sbatch(f"bash run_crr_shift_parallel.sh")
        # slurm.sbatch('python run_crr_shift_parallel.py')
        script_args = [
            dataset,
            alg,
            f'--n_epochs {n_epochs}',
            '--x_opt' if is_x_opt else '',
            f'--cl_min {cl_min}',
            f'--cl_max {cl_max}',
        ]
        if a_min is not None and a_max is not None:
            script_args += [f'--a_min {a_min}', f'--a_max {a_max}']
        slurm.sbatch(f"python run_crr.py {' '.join(script_args)}")

    elif alg.startswith('crr') or alg.startswith('cig'):
        script_args = [
            dataset,
            alg,
            f'--n_epochs {n_epochs}',
            '--x_opt' if is_x_opt else '',
            f'--cl_min {cl_min}',
            f'--cl_max {cl_max}',
        ]
        if alg == 'crr':
            job_name = f'c_{cl_min}_{cl_max}_rr_{dataset}_{n_epochs}'
        elif alg == 'crr_opt':
            job_name = f'c_{cl_min}_{cl_max}_rr_opt_{dataset}_{n_epochs}'
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
                cpus_per_task=50,
                mem='15G',
                qos='cpu-512',
                partition='cpu',
                job_name=job_name,
                output=output_name,
                time=datetime.timedelta(days=0, hours=12, minutes=0, seconds=0),
            )
        slurm.sbatch(f"python run_crr.py {' '.join(script_args)}")
        
    else:
        script_args = [
            dataset,
            alg,
            f'--n_epochs {n_epochs}',
            '--x_opt' if is_x_opt else '',
        ]
        job_name = f'{alg}_{dataset}_{n_epochs}'
        output_name = f'slurm_outputs/{job_name}_{Slurm.JOB_ARRAY_MASTER_ID}_{date}_{h}:{m}:{s}.out'
        if args.gpu:
            slurm = Slurm(
                cpus_per_task=1,
                mem='5G',
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
                mem='5G',
                qos='cpu-4',
                partition='cpu',
                job_name=job_name,
                output=output_name,
                time=datetime.timedelta(days=0, hours=12, minutes=0, seconds=0),
            )
        slurm.sbatch(f"python run_crr.py {' '.join(script_args)}")
