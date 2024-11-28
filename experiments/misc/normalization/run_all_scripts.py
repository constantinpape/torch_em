import os
import shutil
import itertools
import subprocess
from datetime import datetime


def write_batch_script(out_path, _name, dataset, phase, task, norm, dry):
    "Writing scripts for different norm experiments."
    batch_script = f"""#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem 128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete-h100:shared
#SBATCH -G H100:1
#SBATCH -A gzz0001
#SBATCH -c 16
#SBATCH --job-name=unet-{dataset}

source ~/.bashrc
micromamba activate sam \n"""

    # python script
    python_script = "python run_unet.py "

    # dataset choice (livecell / plantseg / mitoem)
    python_script += f"-d {dataset} "

    # phase of code execution (train / predict)
    python_script += f"-p {phase} "

    # nature of task (binary / boundaries)
    python_script += f"-t {task} "

    # normalization scheme (InstanceNorm/OldDefault)
    python_script += f"-n {norm} "

    # let's add the python script to the bash script
    batch_script += python_script

    _op = out_path[:-3] + f"_{os.path.split(_name)[-1]}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    if not dry:
        subprocess.run(["sbatch", _op])


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    script_name = "unet-norm"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")
    return batch_script


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    if args.dataset is None:
        datasets = ["livecell", "plantseg", "mitoem", "gonuclear"]
    else:
        datasets = [args.dataset]

    tasks = ["binary", "boundaries"]
    norms = ["InstanceNormTrackStats", "InstanceNorm"]

    for (dataset, task, norm) in itertools.product(datasets, tasks, norms):
        if dataset == "plantseg" and task == "binary":  # for plantseg: binary is just all pixels as foreground
            continue

        write_batch_script(
            out_path=get_batch_script_names("./gpu_jobs"),
            _name="unet-norm",
            dataset=dataset,
            phase=args.phase,
            task=task,
            norm=norm,
            dry=args.dry,
        )


def main(args):
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-p", "--phase", required=True, type=str)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
