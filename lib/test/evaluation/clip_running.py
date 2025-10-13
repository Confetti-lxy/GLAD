import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Analyzer
import torch


def _save_analyzer_output(seq: Sequence, analyzer: Analyzer, output: dict):
    """Saves the output of the analyzer."""
    if not os.path.exists(analyzer.results_dir):
        print("create analysis result dir:", analyzer.results_dir)
        os.makedirs(analyzer.results_dir)
    if seq.dataset in ['trackingnet', 'got10k']:
        if not os.path.exists(os.path.join(analyzer.results_dir, seq.dataset)):
            os.makedirs(os.path.join(analyzer.results_dir, seq.dataset))
    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(analyzer.results_dir, seq.dataset, seq.name)
    else:
        base_results_path = os.path.join(analyzer.results_dir, seq.name)

    def save_logits(file, data):
        logits = np.array(data).astype(float)
        np.savetxt(file, logits, delimiter='\t', fmt='%.2f')
    
    def save_probs(file, data):
        probs = np.array(data).astype(float)
        np.savetxt(file, probs, delimiter='\t', fmt='%.3f')

    for key, data in output.items():
        # If data is empty
        if not data:
            continue
        
        if key == 'logits':
            logits_file = '{}_logits.txt'.format(base_results_path)
            save_logits(logits_file, data)

        if key == 'probs':
            probs_file = '{}_probs.txt'.format(base_results_path)
            save_probs(probs_file, data)


def run_sequence(seq: Sequence, analyzer: Analyzer, analysis_mode="once", num_gpu=8):
    """Runs a analyzer on a sequence."""
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(analyzer.results_dir, seq.dataset, seq.name)
                bbox_file = '{}_logits.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}_logits.txt'.format(analyzer.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}_logits.txt'.format(analyzer.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist():
        print('Already Analyzed.')
        return

    print('Analyzer: {}, Sequence: {}'.format(analyzer.name, seq.name))

    try:
        if analysis_mode == "once":
            output = analyzer.run_sequence_once(seq)
        else:
            output = analyzer.run_sequence_all(seq)
    except Exception as e:
        print(e)
        return

    sys.stdout.flush()

    print('Analysis Ready.')

    _save_analyzer_output(seq, analyzer, output)


def run_dataset(dataset, analyzers, analysis_mode="once", threads=0, num_gpus=8):
    """Runs a list of analyzers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        analyzers: List of Analyzer instances.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} analyzers on {:5d} sequences'.format(len(analyzers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for analyzer_info in analyzers:
                run_sequence(seq, analyzer_info, analysis_mode)
    elif mode == 'parallel':
        param_list = [(seq, analyzer_info, analysis_mode, num_gpus) for seq, analyzer_info in product(dataset, analyzers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
            pool.terminate()
    print('Done')
