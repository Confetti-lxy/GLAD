import os
import sys
import numpy as np
from lib.test.utils.load_text import load_text
import torch
import pickle
from tqdm import tqdm

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.environment import env_settings


def extract_clip_results(analyzers, dataset, report_name, analysis_mode):
    if analysis_mode == "once":
        return extract_clip_results_once(analyzers, dataset, report_name)
    else:
        return extract_clip_results_all(analyzers, dataset, report_name)


def extract_clip_results_once(analyzers, dataset, report_name):
    settings = env_settings()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    template_better = 0
    search_better = 0
    template_sum_logits = 0
    search_sum_logits = 0
    template_sum_probs = 0
    search_sum_probs = 0
    template_max_logit = -100
    template_min_logit = 100
    search_max_logit = -100
    search_min_logit = 100

    seq_num = len(dataset)

    for seq_id, seq in enumerate(tqdm(dataset)):
        for ana_id, ana in enumerate(analyzers):
            # Load results
            if report_name == "got10k_val":
                base_results_path = '{}/got10k/{}'.format(ana.results_dir, seq.name)
            else:
                base_results_path = '{}/{}'.format(ana.results_dir, seq.name)
            results_path_logits = '{}_logits.txt'.format(base_results_path)
            results_path_probs = '{}_probs.txt'.format(base_results_path)

            if os.path.isfile(results_path_logits):
                logits = torch.tensor(load_text(str(results_path_logits), delimiter=('\t', ','), dtype=np.float64))
            else:
                raise Exception('Result not found. {}'.format(results_path_logits))
            if os.path.isfile(results_path_probs):
                probs = torch.tensor(load_text(str(results_path_probs), delimiter=('\t', ','), dtype=np.float64))
            else:
                raise Exception('Result not found. {}'.format(results_path_probs))

            # Analysis
            if logits[0] >= logits[1]:
                template_better += 1
            else:
                search_better += 1
            
            template_sum_logits += logits[0]
            search_sum_logits += logits[1]
            template_sum_probs += probs[0]
            search_sum_probs += probs[1]
            template_max_logit = max(template_max_logit, logits[0])
            template_min_logit = min(template_min_logit, logits[0])
            search_max_logit = max(search_max_logit, logits[1])
            search_min_logit = min(search_min_logit, logits[1])

    template_average_logits = template_sum_logits / seq_num
    search_average_logits = search_sum_logits / seq_num
    template_average_probs = template_sum_probs / seq_num
    search_average_probs = search_sum_probs / seq_num

    # Prepare dictionary for saving data
    seq_names = [s.name for s in dataset]

    eval_data = {'sequences': seq_names, 'seq_num': seq_num, 'analyzers': ['CLIP'],
                 'template_better': template_better,
                 'search_better': search_better,
                 'template_average_logits': template_average_logits,
                 'template_average_probs': template_average_probs,
                 'template_sum_logits': template_sum_logits,
                 'template_sum_probs': template_sum_probs,
                 'template_max_logit': template_max_logit,
                 'template_min_logit': template_min_logit,
                 'search_average_logits': search_average_logits,
                 'search_average_probs': search_average_probs,
                 'search_sum_logits': search_sum_logits,
                 'search_sum_probs': search_sum_probs,
                 'search_max_logit': search_max_logit,
                 'search_min_logit': search_min_logit,
                 }

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data


def extract_clip_results_all(analyzers, dataset, report_name):
    settings = env_settings()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)
    
    seq_num = len(dataset)

    good_frame_sum = 0
    bad_frame_sum = 0
    good_vid_sum = 0
    bad_vid_sum = 0
    average_template_logit = 0
    average_search_logit = 0
    frame_num = 0

    for seq_id, seq in enumerate(tqdm(dataset)):
        good_frame_num = 0
        bad_frame_num = 0
        average_logit = 0
        for ana_id, ana in enumerate(analyzers):
            # Load results
            if report_name == "got10k_val":
                base_results_path = '{}/got10k/{}'.format(ana.results_dir, seq.name)
            else:
                base_results_path = '{}/{}'.format(ana.results_dir, seq.name)
            results_path_logits = '{}_logits.txt'.format(base_results_path)

            if os.path.isfile(results_path_logits):
                logits = torch.tensor(load_text(str(results_path_logits), delimiter=('\t', ','), dtype=np.float64))
            else:
                raise Exception('Result not found. {}'.format(results_path_logits))

            # Analysis
            # if logits.shape == torch.Size([]):
            #     logits = logits.reshape(1)
            template_logit = logits[0]
            average_template_logit += template_logit
            for search_id, search_logit in enumerate(logits[1:], start=1):
                if search_logit > template_logit:
                    good_frame_num += 1
                else:
                    bad_frame_num += 1
                average_logit += search_logit
            
            good_frame_sum += good_frame_num
            bad_frame_sum += bad_frame_num
            frame_num += len(logits) - 1
            average_search_logit += average_logit
            # if len(logits) == 1:
            #     average_logit = 0
            # else:
            #     average_logit /= len(logits) - 1
            average_logit /= len(logits) - 1
            if average_logit > template_logit:
                good_vid_sum += 1
            else:
                bad_vid_sum += 1
    
    average_template_logit /= seq_num
    average_search_logit /= frame_num

    # Prepare dictionary for saving data
    seq_names = [s.name for s in dataset]

    eval_data = {'sequences': seq_names, 'seq_num': seq_num, 'analyzers': ['CLIP'],
                 'good_frame_sum': good_frame_sum,
                 'bad_frame_sum': bad_frame_sum,
                 'good_vid_sum': good_vid_sum,
                 'bad_vid_sum': bad_vid_sum,
                 'average_template_logit': average_template_logit,
                 'average_search_logit': average_search_logit
                 }

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data


def check_and_load_clip_results(analyzers, dataset, report_name, analysis_mode):
    # Load data
    settings = env_settings()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data_path = os.path.join(result_plot_path, 'eval_data.pkl')

    if os.path.isfile(eval_data_path):
        with open(eval_data_path, 'rb') as fh:
            eval_data = pickle.load(fh)
    else:
        eval_data = extract_clip_results(analyzers, dataset, report_name, analysis_mode)

    return eval_data


def generate_formatted_analyzer_report(row_labels, scores, table_name=''):
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = '{label: <{width}} |'.format(label=table_name, width=name_width)

    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = '{prev} {s: <{width}} |'.format(prev=report_text, s=s, width=s_w)

    report_text = '{prev}\n'.format(prev=report_text)

    for ana_id, d_name in enumerate(row_labels):
        # display name
        report_text = '{prev}{analyzer: <{width}} |'.format(prev=report_text, analyzer=d_name,
                                                           width=name_width)
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = '{prev} {score: <{width}} |'.format(prev=report_text,
                                                              score='{:0.2f}'.format(score_value),
                                                              width=s_w)
        report_text = '{prev}\n'.format(prev=report_text)

    return report_text


def print_clip_results(analyzers, dataset, report_name, analysis_mode):
    """ Print the results for the given analyzers in a formatted table
    args:
        analyzers - List of analyzers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
    """
    # Load pre-computed results
    eval_data = check_and_load_clip_results(analyzers, dataset, report_name, analysis_mode)

    print("{} dataset, {} sequences\n".format(report_name, eval_data['seq_num']))

    if analysis_mode == "all":
        scores = {k: eval_data[k] for k in {'good_frame_sum', 'bad_frame_sum', 'good_vid_sum', 'bad_vid_sum', 'average_template_logit', 'average_search_logit'}}
        scores = dict(sorted(scores.items()))
        # Print
        disp_names = eval_data['analyzers']
        report_text = generate_formatted_analyzer_report(disp_names, scores, table_name=report_name)
        print(report_text)
        return

    scores_overall = {k: eval_data[k] for k in {'template_better', 'search_better'}}
    scores_overall = dict(sorted(scores_overall.items()))
    scores_template = {k.replace('template_', ''): eval_data[k] for k in {'template_average_logits', 
                                                 'template_average_probs', 
                                                 'template_sum_logits', 
                                                 'template_sum_probs', 
                                                 'template_max_logit', 
                                                 'template_min_logit'}}
    scores_template = dict(sorted(scores_template.items()))
    scores_search = {k.replace('search_', ''): eval_data[k] for k in {'search_average_logits', 
                                               'search_average_probs', 
                                               'search_sum_logits', 
                                               'search_sum_probs', 
                                               'search_max_logit', 
                                               'search_min_logit'}}
    scores_search = dict(sorted(scores_search.items()))

    # Print
    disp_names = eval_data['analyzers']
    report_text_overall = generate_formatted_analyzer_report(disp_names, scores_overall, table_name=report_name)
    report_text_template = generate_formatted_analyzer_report(disp_names, scores_template, table_name=report_name)
    report_text_search = generate_formatted_analyzer_report(disp_names, scores_search, table_name=report_name)
    print("Overall:")
    print(report_text_overall)
    print("Template:")
    print(report_text_template)
    print("Search:")
    print(report_text_search)