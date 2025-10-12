import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.clip_running import run_dataset
from lib.test.evaluation.clip_analyzer import Analyzer


def run_analyzer(analyzer_name, dataset_name='otb', sequence=None, analysis_mode="once", threads=0, num_gpus=8):
    """Run analyzer on sequence or dataset.
    args:
        analyzer_name: Name of tracking method.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    analyzers = [Analyzer(analyzer_name, dataset_name)]

    run_dataset(dataset, analyzers, analysis_mode, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run analyzer on sequence or dataset.')
    parser.add_argument('analyzer_name', type=str, default='CLIP', help='Name of analysis method.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--analysis_mode', type=str, default="once", help="Select one frame or all frames.")
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_analyzer(args.analyzer_name, args.dataset_name, seq_name, args.analysis_mode, args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()
