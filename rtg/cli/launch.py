#!/usr/bin/env python

import os
import subprocess
import sys
from argparse import REMAINDER, ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List, Tuple

import torch
from rtg import log

# adapted from torch.distributed.launch


def parse_args(args=None):
    parser = ArgumentParser(
        description="PyTorch distributed training launch helper utilty that will spawn up "
        "multiple distributed processes",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Optional arguments for the launch helper
    parser.add_argument(
        "-N",
        "--nodes",
        metavar='INT',
        type=int,
        default=1,
        help="The number of nodes to use for distributed " "training",
    )
    parser.add_argument(
        "-r",
        "--node-rank",
        metavar='INT',
        type=int,
        default=0,
        help="The rank of the node for multi-node distributed " "training",
    )

    parser.add_argument(
        "-P",
        "--procs-per-node",
        metavar='INT',
        type=int,
        default=1,
        help="The number of processes to launch on each node with one gpu each, "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.",
    )

    parser.add_argument(
        "-G",
        "--gpus-per-proc",
        metavar='INT',
        type=int,
        default=0,
        help="Number of GPUs to assign to each process. ",
    )
    parser.add_argument(
        "--master-addr",
        metavar='HostOrIP',
        default="127.0.0.1",
        type=str,
        help="Master node (rank 0)'s address, should be either "
        "the IP address or the hostname of node 0, for "
        "single node multi-proc training, the "
        "--master_addr can simply be 127.0.0.1",
    )
    parser.add_argument(
        "--master-port",
        metavar="Port",
        default=29500,
        type=int,
        help="Master node (rank 0)'s free port that needs to "
        "be used for communciation during distributed "
        "training",
    )

    parser.add_argument(
        "-m",
        "--module",
        dest='is_module',
        default=False,
        action="store_true",
        help="Treats the <script> argument as python module and executes CLI with 'python -m <script>'.",
    )
    parser.add_argument(
        "--stdin",
        dest='stdin',
        default=False,
        action="store_true",
        help="Input is from stdin. Each line is distributed to a process in round-robin fashion.",
    )
    # positional
    parser.add_argument(
        "script",
        type=str,
        default='rtg-pipeline',
        help="The full path to the training script (or qualified module name if -m/--module) "
        "to be launched in parallel, followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument('script_args', help="arguments to script or module", nargs=REMAINDER)
    return parser.parse_args(args)


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
        log.info(
            "*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process to be"
            f" {cur_env['OMP_NUM_THREADS']} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************"
        )

    avail_gpus = torch.cuda.device_count()
    assum_gpus = args.procs_per_node * args.gpus_per_proc
    if avail_gpus < assum_gpus:
        import socket

        msg = (
            f'Host={socket.gethostname()}; GPUs available={avail_gpus}; requested={assum_gpus}'
            f'\n\tprocs-per-node * gpus-per-proc = {args.procs_per_node} * {args.gpus_per_proc}'
        )
        raise Exception(msg)
    if avail_gpus > 0 >= assum_gpus:
        log.warning("WARNING: GPUs are available but the --gpus-per-proc is not set.")

    cmd = []
    if args.is_module:
        cmd += [sys.executable, "-m"]
    cmd.append(args.script)
    cmd.extend(args.script_args)
    log.info(f'{cmd}')
    processes = []
    STDIN = subprocess.PIPE if args.stdin else subprocess.DEVNULL
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        device_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        device_ids = [str(i) for i in range(torch.cuda.device_count())]
    n_required_gpus = args.procs_per_node * args.gpus_per_proc
    assert (
        len(device_ids) >= n_required_gpus
    ), f'Visible GPUs={len(device_ids)}; Required={n_required_gpus} (i.e {args.procs_per_node} * {args.gpus_per_proc}))'

    for local_rank in range(0, args.procs_per_node):
        my_env = cur_env.copy()
        # each process's rank
        dist_rank = args.procs_per_node * args.node_rank + local_rank
        my_env["RANK"] = str(dist_rank)
        my_env["LOCAL_RANK"] = str(local_rank)
        my_env['RTG_PROC_NAME'] = f'r{dist_rank}'
        if args.gpus_per_proc > 0:
            device_idx = list(range(local_rank * args.gpus_per_proc, (local_rank + 1) * args.gpus_per_proc))
            my_device_ids = ','.join(device_ids[idx] for idx in device_idx)
            my_env["CUDA_VISIBLE_DEVICES"] = my_device_ids
            my_env['RTG_PROC_NAME'] += f'.dev{my_device_ids}'
        # spawn processes
        process = subprocess.Popen(cmd, env=my_env, shell=False, stdin=STDIN, text=True, cwd=os.getcwd())
        processes.append(process)

    if args.stdin:
        distribute_stdin(processes)
    else:
        wait_till_end(processes, cmd=cmd)


def distribute_stdin(processes: List[subprocess.Popen], method="round-robin"):
    assert method == "round-robin"  # only this is supported for now
    alive_procs = processes
    dead_procs = []
    try:
        for i, line in enumerate(sys.stdin):
            if not alive_procs:
                break
            proc = processes[i % len(alive_procs)]
            try:
                # any encoding errors are replaced with '?'
                line = line.encode(encoding='utf-8', errors='replace').decode('utf-8')
                proc.stdin.write(line)
            except BrokenPipeError:
                log.warning(f"Process {proc.pid} is dead. STDIN Line dropped")
                dead_procs.append(proc)
                alive_procs.remove(proc)
    finally:
        teardown(processes)


def wait_till_end(processes: List[subprocess.Popen], cmd, timeout: int = 10):
    log.info(f"Launched {len(processes)} processes")
    alive = [True] * len(processes)
    try:
        while sum(alive) > 0:
            for i, process in enumerate(processes):
                try:
                    process.wait(timeout=timeout)
                    alive[i] = False
                    if process.returncode != 0:
                        # TODO: communicate to all nodes
                        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
                except subprocess.TimeoutExpired:
                    pass  # that's okay! skip this and check on next process
        log.info('All processes completed successfully')
    finally:
        teardown(processes)


def teardown(procs: List[subprocess.Popen]):
    errs = []
    for proc in procs:  # force abort any running processes
        try:
            if proc.poll() is None:  # process is still running
                log.warning(f"Force terminating process: {proc.pid}")
                proc.terminate()
        except:
            errs.append(proc.pid)
    if errs:
        errs = ' '.join(str(x) for x in errs)
        log.warning(f"Error terminating some process(es). Please run 'kill -9 {errs}'")


if __name__ == "__main__":
    main()