#!/usr/bin/env python

import os
import subprocess
import sys
from argparse import ArgumentParser, REMAINDER, ArgumentDefaultsHelpFormatter as HelpFormatter
from typing import List, Tuple

import torch


# adapted from torch.distributed.launch

def parse_args():
    parser = ArgumentParser(
        description="PyTorch distributed training launch helper utilty that will spawn up "
                    "multiple distributed processes",
        formatter_class=HelpFormatter)

    # Optional arguments for the launch helper
    parser.add_argument("-N", "--nodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("-r", "--node-rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")

    parser.add_argument("-P", "--procs-per-node", type=int, default=1,
                        help="The number of processes to launch on each node with one gpu each, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")

    parser.add_argument("-G", "--gpus-per-proc", type=int, default=0,
                        help="Number of GPUs to assign to each process. ")
    parser.add_argument("--master-addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master-port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    # exclusive
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-m", "--module", default=False, action="store_true",
                       help="Changes each process to interpret the launch script "
                            "as a python module, executing with the same behavior as"
                            "'python -m'.")
    group.add_argument("--no_python", default=False, action="store_true",
                       help="Do not prepend the training script with \"python\" - just exec "
                            "it directly. Useful when the script is not a Python script.")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main(args=None):
    args = args or parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nodes * args.procs_per_node

    # set PyTorch distributed related environmental variables
    cur_env = os.environ.copy()
    cur_env["MASTER_ADDR"] = args.master_addr
    cur_env["MASTER_PORT"] = str(args.master_port)
    cur_env["WORLD_SIZE"] = str(dist_world_size)

    if 'OMP_NUM_THREADS' not in os.environ and args.procs_per_node > 1:
        cur_env["OMP_NUM_THREADS"] = str(2)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process to be"
              f" {cur_env['OMP_NUM_THREADS']} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************", file=sys.stderr)

    avail_gpus = torch.cuda.device_count()
    assum_gpus = args.procs_per_node * args.gpus_per_proc
    if avail_gpus < assum_gpus:
        import socket
        msg = f'Host={socket.gethostname()}; GPUs available={avail_gpus}; requested={assum_gpus}' \
              f'\n\tprocs-per-node * gpus-per-proc = {args.procs_per_node} * {args.gpus_per_proc}'

        raise Exception(msg)
    if avail_gpus > 0 >= assum_gpus:
        print("WARNING: GPUs are available but the --gpus-per-proc is not set.", file=sys.stderr)

    with_python = not args.no_python
    cmd = []
    if with_python:
        cmd += [sys.executable, "-u"]
        if args.module:
            cmd.append("-m")
    else:
        if args.module:
            raise ValueError(
                "Don't use both the '--no_python' flag and the '--module' flag at the same time.")

    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    processes = []
    for local_rank in range(0, args.procs_per_node):
        my_env = cur_env.copy()
        # each process's rank
        dist_rank = args.procs_per_node * args.node_rank + local_rank
        my_env["RANK"] = str(dist_rank)
        my_env["LOCAL_RANK"] = str(local_rank)
        if args.gpus_per_proc > 0:
            dev_ids = range(local_rank * args.gpus_per_proc, (local_rank + 1) * args.gpus_per_proc)
            device_ids = ','.join(str(i) for i in dev_ids)
            my_env["CUDA_VISIBLE_DEVICES"] = device_ids

        # spawn the processes
        process = subprocess.Popen(cmd, env=my_env)
        processes.append((cmd, process))

    timeout = 10
    alive = [True] * len(processes)
    try:
        while sum(alive) > 0:
            for i, (cmd, process) in enumerate(processes):
                try:
                    process.wait(timeout=timeout)
                    alive[i] = False
                    if process.returncode != 0:
                        # TODO: communicate to all nodes
                        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
                except subprocess.TimeoutExpired:
                    pass  # that's okay! skip this and check on next process
    finally:
        # kill all living processes
        teardown([proc for is_alive, (cmd, proc) in zip(alive, processes) if is_alive])


def teardown(procs: List[subprocess.Popen]):
    errs = []
    for proc in procs:  # force abort any running processes
        try:
            if proc.poll() is None:  # process is still running
                print(f"Force terminating process: {proc.pid}", file=sys.stderr)
                proc.terminate()
        except:
            errs.append(proc.pid)
    if errs:
        errs = ' '.join(str(x) for x in errs)
        print(f"Error terminating some process(es). Please run 'kill -9 {errs}'", file=sys.stderr)


if __name__ == "__main__":
    main()
