"""torchrun entry point for test_ep.py (multi-node without nested spawn).

`test_ep.py` launches its ranks with `torch.multiprocessing.spawn` and
expects the mp.spawn env convention (see `deep_ep.utils.envs.init_dist`):
`WORLD_SIZE` = number of NODES and `RANK` = node rank. That works well with
one process per node (e.g. the Slurm launchers in awsome-distributed-ai),
but under `torchrun` -- the launcher most Kubernetes and multi-node setups
already use -- it fails in two non-obvious ways:

1. torchrun sets `WORLD_SIZE` to the TOTAL process count and `RANK` to the
   global rank. Passed through to `init_dist`, every rank waits for
   `nodes * procs-per-node` peers that will never exist (observed: a 2x8
   launch stalls forever waiting for 64 ranks).
2. torchrun sets `TORCHELASTIC_USE_AGENT_STORE=1`, making every rank a
   TCPStore CLIENT of the rendezvous agent's store. If `MASTER_PORT` does
   not equal the rendezvous port, no process binds the port and all ranks
   spin in connect-retry (observed: 16 processes parked in
   hrtimer_nanosleep with no listener).

This wrapper normalizes the environment BEFORE importing test_ep, then
calls `test_loop` directly (torchrun already provides the process fan-out,
so nested mp.spawn is unnecessary and wasteful). All test_ep.py CLI flags
are accepted unchanged, except `--num-processes`, which is ignored in
favor of torchrun's `--nproc-per-node`.

Example (2 nodes x 8 GPUs, run once per node):

    torchrun --nnodes=2 --node-rank=<0|1> --nproc-per-node=8 \
        --rdzv-backend=c10d --rdzv-endpoint=<node0-ip>:29650 \
        tests/elastic/test_ep_torchrun.py --num-tokens 128 --num-topk 8

The rendezvous port doubles as MASTER_PORT (trap 2); override with
--master-port only if your rendezvous endpoint uses a different port than
the TCPStore should.
"""
import os

from test_ep import build_parser, test_loop


def main() -> None:
    parser = build_parser()
    parser.add_argument('--master-port', type=int, default=0,
                        help='TCPStore port; defaults to the torchrun rendezvous port')
    args = parser.parse_args()

    local_world = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    node_rank = int(os.environ.get('GROUP_RANK',
                                   str(int(os.environ['RANK']) // local_world)))
    num_nodes = int(os.environ.get('GROUP_WORLD_SIZE',
                                   str(int(os.environ['WORLD_SIZE']) // local_world)))

    # Trap 1: init_dist reads WORLD_SIZE/RANK in the mp.spawn convention
    # (nodes / node rank), not torchrun's (total procs / global rank).
    os.environ['WORLD_SIZE'] = str(num_nodes)
    os.environ['RANK'] = str(node_rank)

    # Trap 2: under the torchelastic agent store every rank is a TCPStore
    # client -- MASTER_PORT must be a port something actually binds, which
    # is the rendezvous port unless the user says otherwise.
    if args.master_port:
        os.environ['MASTER_PORT'] = str(args.master_port)
    elif 'TORCHELASTIC_USE_AGENT_STORE' in os.environ:
        rdzv_port = os.environ.get('MASTER_PORT')
        if rdzv_port is None:
            raise RuntimeError(
                'MASTER_PORT is unset; pass --master-port or use '
                '--rdzv-backend=c10d (which sets it)')

    if args.num_processes != local_world:
        print(f'[test_ep_torchrun] ignoring --num-processes={args.num_processes}; '
              f'torchrun provides {local_world} local ranks', flush=True)

    test_loop(local_rank, local_world, args)


if __name__ == '__main__':
    main()
