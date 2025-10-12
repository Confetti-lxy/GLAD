import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.extract_clip_results import print_clip_results
from lib.test.evaluation import get_dataset
from lib.test.evaluation.clip_analyzer import Analyzer
import argparse


parser = argparse.ArgumentParser(description='Run analyzer on sequence or dataset.')
parser.add_argument('analyzer_name', type=str, default='CLIP', help='Name of analysis method.')
parser.add_argument('--dataset_name', type=str, help='Name of config file.')
parser.add_argument('--analysis_mode', type=str, default="once", help="Select one frame or all frames.")
args = parser.parse_args()

dataset_name = args.dataset_name
analyzers = [Analyzer(args.analyzer_name, dataset_name)]

print(f"Preparing dataset {dataset_name}...")
dataset = get_dataset(dataset_name)
print("Done!")
print_clip_results(analyzers, dataset, dataset_name, args.analysis_mode)
