# Script which was used to generate the figures in the EpiK-Eval paper: https://arxiv.org/abs/2310.15372

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def string_to_list(s):
    # Convert the string to a dictionary
    counts_dict = {int(k): int(v) for k, v in (pair.split(":") for pair in s.split(","))}

    # Convert the dictionary to a distribution
    distribution = []
    for k, v in counts_dict.items():
        distribution.extend([k] * v)
    return distribution


def plot_accuracy(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented, 
                  t5_sizes, opt_sizes,
                  filename):
    plt.close()

    plt.plot(t5_sizes, t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    plt.plot(t5_sizes, t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    plt.plot(t5_sizes, flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    plt.plot(t5_sizes, flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    plt.plot(opt_sizes, opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    plt.plot(opt_sizes, opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')

    plt.xlabel('Parameters', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=12, length=5)
    plt.tick_params(axis='x', which='minor', length=2)
    plt.ylim([-1, 101])
    plt.tight_layout()
    model_legend = plt.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10),
                               Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10),
                               Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10)],
                              ['T5', 'Flan-T5', 'OPT'],
                              loc=2,
                              fontsize=11)
    plt.legend([Line2D([0], [0], color='#007fff', linestyle='solid', linewidth=2),
                Line2D([0], [0], color='#ff9800', linestyle='dotted', linewidth=2)],
               ['unsegmented', 'segmented'],
               loc=1,
               fontsize=11)
    plt.gca().add_artist(model_legend)
    plt.savefig(filename)


def plot_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            filename):
    plt.close()

    fig, axarr = plt.subplots(1, 3, figsize=(6.4 * 3, 4.8), sharey=True)

    axarr[0].plot(t5_sizes, recall_t5_unsegmented, marker='o', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(t5_sizes, recall_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(t5_sizes, recall_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(t5_sizes, recall_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(opt_sizes, recall_opt_unsegmented, marker='s', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(opt_sizes, recall_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].set_xscale('log')
    axarr[0].set_title("Recall", fontsize=22)
    axarr[0].set_xlabel("Parameters", fontsize=20)
    axarr[0].set_ylabel("Accuracy (%)", fontsize=20)
    axarr[0].set_ylim([-1, 101])
    axarr[0].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[0].tick_params(axis='x', which='minor', length=2.8)

    axarr[1].plot(t5_sizes, reasoning_t5_unsegmented, marker='o', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(t5_sizes, reasoning_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(t5_sizes, reasoning_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(t5_sizes, reasoning_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(opt_sizes, reasoning_opt_unsegmented, marker='s', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(opt_sizes, reasoning_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].set_xscale('log')
    axarr[1].set_title("Reasoning", fontsize=22)
    axarr[1].set_xlabel("Parameters", fontsize=20)
    axarr[1].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[1].tick_params(axis='x', which='minor', length=2.8)

    axarr[2].plot(t5_sizes, answer_t5_unsegmented, marker='o', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[2].plot(t5_sizes, answer_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[2].plot(t5_sizes, answer_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[2].plot(t5_sizes, answer_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[2].plot(opt_sizes, answer_opt_unsegmented, marker='s', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[2].plot(opt_sizes, answer_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[2].set_xscale('log')
    axarr[2].set_title("Final Answer", fontsize=22)
    axarr[2].set_xlabel("Parameters", fontsize=20)
    axarr[2].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[2].tick_params(axis='x', which='minor', length=2.8)

    model_legend = fig.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=13),
                               Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=13),
                               Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=13)],
                              ['T5', 'Flan-T5', 'OPT'],
                              bbox_to_anchor=(0.992, 0.6),
                              fontsize=16)
    fig.legend([Line2D([0], [0], color='#007fff', linestyle='solid', linewidth=2.7),
                Line2D([0], [0], color='#ff9800', linestyle='dotted', linewidth=2.7)],
               ['unsegmented', 'segmented'],
               bbox_to_anchor=(0.992, 0.37),
               fontsize=16)
    fig.gca().add_artist(model_legend)

    plt.tight_layout()
    plt.savefig(filename)


def plot_hallucination_rate(train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                            test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                            t5_sizes, opt_sizes,
                            filename):
    plt.close()

    fig, axarr = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8), sharey=True)

    axarr[0].plot(t5_sizes, train_t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    axarr[0].plot(t5_sizes, train_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(t5_sizes, train_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    axarr[0].plot(t5_sizes, train_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(opt_sizes, train_opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    axarr[0].plot(opt_sizes, train_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].set_xscale('log')
    axarr[0].set_title("Train", fontsize=16)
    axarr[0].set_xlabel("Parameters", fontsize=14)
    axarr[0].set_ylabel("Hallucination Rate (%)", fontsize=14)
    axarr[0].set_ylim([-1, 101])
    axarr[0].tick_params(axis='both', which='major', labelsize=12, length=5)
    axarr[0].tick_params(axis='x', which='minor', length=2)

    axarr[1].plot(t5_sizes, test_t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    axarr[1].plot(t5_sizes, test_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(t5_sizes, test_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    axarr[1].plot(t5_sizes, test_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(opt_sizes, test_opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    axarr[1].plot(opt_sizes, test_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].set_xscale('log')
    axarr[1].set_title("Test", fontsize=16)
    axarr[1].set_xlabel("Parameters", fontsize=14)
    axarr[1].tick_params(axis='both', which='major', labelsize=12, length=5)
    axarr[1].tick_params(axis='x', which='minor', length=2)

    model_legend = fig.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10),
                               Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10),
                               Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10)],
                              ['T5', 'Flan-T5', 'OPT'],
                              bbox_to_anchor=(0.163, 0.807),
                              fontsize=11)
    fig.legend([Line2D([0], [0], color='#007fff', linestyle='solid', linewidth=2),
                Line2D([0], [0], color='#ff9800', linestyle='dotted', linewidth=2)],
               ['unsegmented', 'segmented'],
               bbox_to_anchor=(0.202, 0.92),
               fontsize=11)
    fig.gca().add_artist(model_legend)

    plt.tight_layout()
    plt.savefig(filename)


def plot_answer_length_distribution(t5_unsegmented_distribution, t5_segmented_distribution, 
                                    flan_unsegmented_distribution, flan_segmented_distribution, 
                                    opt_unsegmented_distribution, opt_segmented_distribution,
                                    target_distribution,
                                    filename):
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(6.4 * 3, 4.8), sharey=True)  # 3 subplots arranged horizontally

    bins = np.arange(1, 9)  # We want bins for each integer from 1 to 7, thus the 9 here.

    models = ['T5-XL', 'Flan-T5-XL', 'OPT-2.7B']
    distributions_unsegmented = [t5_unsegmented_distribution, flan_unsegmented_distribution, opt_unsegmented_distribution]
    distributions_segmented = [t5_segmented_distribution, flan_segmented_distribution, opt_segmented_distribution]

    for ax, model, distribution_unsegmented, distribution_segmented in zip(axes, models, distributions_unsegmented, distributions_segmented):
        ax.hist(distribution_unsegmented, bins=bins, alpha=0.33, label='Unsegmented')
        ax.hist(distribution_segmented, bins=bins, alpha=0.33, label='Segmented')
        ax.hist(target_distribution, bins=bins, alpha=0.33, label='Target')
        
        ax.set_xlabel('Number of Sentences', fontsize=20)
        ax.set_xticks(bins[:-1])  # Set x-ticks to be the integer values
        ax.tick_params(axis='both', which='major', labelsize=16, length=7)
        ax.set_title(model, fontsize=22)
    axes[0].set_ylabel('Count', fontsize=20)

    axes[2].legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(filename)


def plot_task_whole_accuracy(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented, 
                             t5_sizes, opt_sizes,
                             filename):
    plt.close()

    plt.plot(t5_sizes, t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    plt.plot(t5_sizes, t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    plt.plot(t5_sizes, flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    plt.plot(t5_sizes, flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    plt.plot(opt_sizes, opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    plt.plot(opt_sizes, opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')

    plt.title("Whole Answer", fontsize=22)
    plt.xlabel('Parameters', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=16, length=7)
    plt.tick_params(axis='x', which='minor', length=2.8)
    plt.ylim([-1, 101])
    plt.tight_layout()
    plt.savefig(filename)
    
def plot_task_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                                 reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                                 answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                                 t5_sizes, opt_sizes,
                                 filename):
    
    plt.close()

    fig, axarr = plt.subplots(1, 3, figsize=(6.4 * 3, 4.8), sharey=True)

    axarr[0].plot(t5_sizes, recall_t5_unsegmented, marker='o', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(t5_sizes, recall_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(t5_sizes, recall_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(t5_sizes, recall_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(opt_sizes, recall_opt_unsegmented, marker='s', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(opt_sizes, recall_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].set_xscale('log')
    axarr[0].set_title("Recall", fontsize=22)
    axarr[0].set_xlabel("Parameters", fontsize=20)
    axarr[0].set_ylabel("Accuracy (%)", fontsize=20)
    axarr[0].set_ylim([-1, 101])
    axarr[0].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[0].tick_params(axis='x', which='minor', length=2.8)

    axarr[1].plot(t5_sizes, reasoning_t5_unsegmented, marker='o', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(t5_sizes, reasoning_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(t5_sizes, reasoning_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(t5_sizes, reasoning_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(opt_sizes, reasoning_opt_unsegmented, marker='s', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(opt_sizes, reasoning_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].set_xscale('log')
    axarr[1].set_title("Reasoning", fontsize=22)
    axarr[1].set_xlabel("Parameters", fontsize=20)
    axarr[1].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[1].tick_params(axis='x', which='minor', length=2.8)

    axarr[2].plot(t5_sizes, answer_t5_unsegmented, marker='o', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[2].plot(t5_sizes, answer_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[2].plot(t5_sizes, answer_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[2].plot(t5_sizes, answer_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[2].plot(opt_sizes, answer_opt_unsegmented, marker='s', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[2].plot(opt_sizes, answer_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[2].set_xscale('log')
    axarr[2].set_title("Final Answer", fontsize=22)
    axarr[2].set_xlabel("Parameters", fontsize=20)
    axarr[2].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[2].tick_params(axis='x', which='minor', length=2.8)

    plt.tight_layout()
    plt.savefig(filename)

def plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                                              answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                                              t5_sizes, opt_sizes,
                                              filename):
    plt.close()

    fig, axarr = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8), sharey=True)

    axarr[0].plot(t5_sizes, recall_t5_unsegmented, marker='o', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(t5_sizes, recall_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(t5_sizes, recall_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(t5_sizes, recall_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(opt_sizes, recall_opt_unsegmented, marker='s', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[0].plot(opt_sizes, recall_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].set_xscale('log')
    axarr[0].set_title("Recall", fontsize=22)
    axarr[0].set_xlabel("Parameters", fontsize=20)
    axarr[0].set_ylabel("Accuracy (%)", fontsize=20)
    axarr[0].set_ylim([-1, 101])
    axarr[0].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[0].tick_params(axis='x', which='minor', length=2.8)

    axarr[1].plot(t5_sizes, answer_t5_unsegmented, marker='o', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(t5_sizes, answer_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(t5_sizes, answer_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(t5_sizes, answer_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(opt_sizes, answer_opt_unsegmented, marker='s', linestyle='solid', linewidth=0.8, color='#007fff')
    axarr[1].plot(opt_sizes, answer_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].set_xscale('log')
    axarr[1].set_title("Final Answer", fontsize=22)
    axarr[1].set_xlabel("Parameters", fontsize=20)
    axarr[1].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[1].tick_params(axis='x', which='minor', length=2.8)

    plt.tight_layout()
    plt.savefig(filename)
    
def plot_task_hallucination_rate(train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                 test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                 t5_sizes, opt_sizes,
                                 filename):
    plt.close()

    fig, axarr = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8), sharey=True)

    axarr[0].plot(t5_sizes, train_t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    axarr[0].plot(t5_sizes, train_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(t5_sizes, train_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    axarr[0].plot(t5_sizes, train_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].plot(opt_sizes, train_opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    axarr[0].plot(opt_sizes, train_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[0].set_xscale('log')
    axarr[0].set_title("Train", fontsize=22)
    axarr[0].set_xlabel("Parameters", fontsize=20)
    axarr[0].set_ylabel("Hallucination Rate (%)", fontsize=20)
    axarr[0].set_ylim([-1, 101])
    axarr[0].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[0].tick_params(axis='x', which='minor', length=2.8)

    axarr[1].plot(t5_sizes, test_t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    axarr[1].plot(t5_sizes, test_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(t5_sizes, test_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    axarr[1].plot(t5_sizes, test_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].plot(opt_sizes, test_opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    axarr[1].plot(opt_sizes, test_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    axarr[1].set_xscale('log')
    axarr[1].set_title("Test", fontsize=22)
    axarr[1].set_xlabel("Parameters", fontsize=20)
    axarr[1].tick_params(axis='both', which='major', labelsize=16, length=7)
    axarr[1].tick_params(axis='x', which='minor', length=2.8)

    plt.tight_layout()
    plt.savefig(filename)
    
def plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented, 
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    filename):
    plt.close()
    
    fig = plt.figure(figsize=(6.4 * 3, 4.8))
    gs = plt.GridSpec(1, 5, width_ratios=[6.4, -1.15, 6.4, -2.15, 6.4])

    ax0 = fig.add_subplot(gs[0, 0])  # First plot
    ax1 = fig.add_subplot(gs[0, 2])  # Second plot
    ax2 = fig.add_subplot(gs[0, 4]) # Third plot

    ax0.plot(t5_sizes, t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    ax0.plot(t5_sizes, t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    ax0.plot(t5_sizes, flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    ax0.plot(t5_sizes, flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    ax0.plot(opt_sizes, opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    ax0.plot(opt_sizes, opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')

    ax0.set_title("Whole Answer", fontsize=22)
    ax0.set_xlabel('Parameters', fontsize=20)
    ax0.set_ylabel('Accuracy (%)', fontsize=20)
    ax0.set_xscale('log')
    ax0.tick_params(axis='both', which='major', labelsize=16, length=7)
    ax0.tick_params(axis='x', which='minor', length=2.8)
    ax0.set_ylim([-1, 101])

    ax1.plot(t5_sizes, train_t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    ax1.plot(t5_sizes, train_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    ax1.plot(t5_sizes, train_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    ax1.plot(t5_sizes, train_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    ax1.plot(opt_sizes, train_opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    ax1.plot(opt_sizes, train_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    ax1.set_xscale('log')
    ax1.set_title("Train", fontsize=22)
    ax1.set_xlabel("Parameters", fontsize=20)
    ax1.set_ylabel("Hallucination Rate (%)", fontsize=20)
    ax1.set_ylim([-1, 101])
    ax1.tick_params(axis='both', which='major', labelsize=16, length=7)
    ax1.tick_params(axis='x', which='minor', length=2.8)

    ax2.plot(t5_sizes, test_t5_unsegmented, marker='o', linestyle='solid', linewidth=1, color='#007fff')
    ax2.plot(t5_sizes, test_t5_segmented, marker='o', linestyle='dotted', linewidth=1, color='#ff9800')
    ax2.plot(t5_sizes, test_flan_t5_unsegmented, marker='^', linestyle='solid', linewidth=1, color='#007fff')
    ax2.plot(t5_sizes, test_flan_t5_segmented, marker='^', linestyle='dotted', linewidth=1, color='#ff9800')
    ax2.plot(opt_sizes, test_opt_unsegmented, marker='s', linestyle='solid', linewidth=1, color='#007fff')
    ax2.plot(opt_sizes, test_opt_segmented, marker='s', linestyle='dotted', linewidth=1, color='#ff9800')
    ax2.sharey(ax1)
    ax2.yaxis.set_tick_params(labelleft=False)
    ax2.set_xscale('log')
    ax2.set_title("Test", fontsize=22)
    ax2.set_xlabel("Parameters", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16, length=7)
    ax2.tick_params(axis='x', which='minor', length=2.8)

    plt.tight_layout()
    plt.savefig(filename)


def plot_legend(filename):
    plt.close()

    fig, ax = plt.subplots(figsize=(4,2))
    leg = ax.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=13),
                    Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=13),
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=13),
                    Line2D([0], [0], color='#007fff', linestyle='solid', linewidth=2.7),
                    Line2D([0], [0], color='#ff9800', linestyle='dotted', linewidth=2.7)],
                    ['T5', 'Flan-T5', 'OPT', 'unsegmented', 'segmented'],
                    loc='center',
                    ncol=5,
                    fontsize=16)
    ax.axis('off')

    # Ensure the figure only contains the legend by making the legend's bounding box fill the figure
    fig.canvas.draw()  # Necessary to get the bbox (bounding box) correctly
    bb = leg.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(bb.width, bb.height)

    fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)


def main():
    # Accuracy
    t5_sizes = [80000000, 250000000, 780000000, 3000000000]
    opt_sizes = [125000000, 350000000, 1300000000, 2700000000]
    
    t5_unsegmented = [58.00, 66.56, 68.00, 62.00]
    t5_segmented = [5.78, 9.67, 10.44, 15.67]
    flan_t5_unsegmented = [64.56, 68.56, 73.11, 77.33]
    flan_t5_segmented = [5.67, 8.56, 8.89, 14.56]
    opt_unsegmented = [46.89, 53.44, 57.78, 65.44]
    opt_segmented = [7.00, 7.33, 12.22, 12.89]

    plot_accuracy(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented, 
                  t5_sizes, opt_sizes,
                  '../figures/accuracy.pdf')
    
    # Accuracy Breakdown
    recall_t5_unsegmented = [88.22, 89.78, 90.44, 93.22]
    recall_t5_segmented = [7.22, 11.56, 11.89, 20.78]
    recall_flan_t5_unsegmented = [86.67, 89.89, 87.78, 91.44]
    recall_flan_t5_segmented = [7.11, 10.00, 9.44, 16.78]
    recall_opt_unsegmented = [62.11, 73.67, 74.89, 78.33]
    recall_opt_segmented = [8.22, 8.56, 13.78, 13.89]

    reasoning_t5_unsegmented = [72.43, 76.89, 76.36, 63.57]
    reasoning_t5_segmented = [84.85, 89.74, 90.24, 73.02]
    reasoning_flan_t5_unsegmented = [86.11, 83.85, 82.95, 84.87]
    reasoning_flan_t5_segmented = [89.29, 89.47, 93.10, 81.69]
    reasoning_opt_unsegmented = [85.26, 80.97, 85.02, 87.12]
    reasoning_opt_segmented = [94.29, 88.89, 88.89, 92.19]

    answer_t5_unsegmented = [72.60, 80.19, 81.71, 75.71]
    answer_t5_segmented = [86.67, 87.00, 91.26, 82.94]
    answer_flan_t5_unsegmented = [77.99, 80.44, 88.20, 89.00]
    answer_flan_t5_segmented = [83.61, 89.53, 96.39, 94.93]
    answer_opt_unsegmented = [79.47, 77.58, 81.25, 87.26]
    answer_opt_segmented = [87.50, 89.19, 93.22, 96.67]

    plot_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            '../figures/accuracy_breakdown.pdf')
    
    # Hallucination Rate
    train_t5_unsegmented = [0.03, 0.04, 0.03, 0.03]
    train_t5_segmented = [0, 0, 0, 0.01]
    train_flan_t5_unsegmented = [0.03, 0.03, 0.03, 0.03]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0.19, 0]
    train_opt_segmented = [0, 0.04, 0, 0]

    test_t5_unsegmented = [5.30, 4.07, 3.58, 1.94]
    test_t5_segmented = [56.92, 53.41, 52.68, 41.04]
    test_flan_t5_unsegmented = [6.06, 4.53, 4.77, 2.85]
    test_flan_t5_segmented = [59.93, 55.59, 51.28, 43.63]
    test_opt_unsegmented = [16.40, 9.05, 9.45, 8.52]
    test_opt_segmented = [56.36, 57.67, 49.33, 47.91]

    plot_hallucination_rate(train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                            test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                            t5_sizes, opt_sizes,
                            '../figures/hallucination_rate.pdf')
    
    # Answer Length Distribution
    t5_unsegmented_distribution = string_to_list("1: 14, 2: 50, 3: 265, 4: 366, 5: 178, 6: 27")
    t5_segmented_distribution = string_to_list("1: 7, 2: 50, 3: 238, 4: 317, 5: 233, 6: 55")
    flan_unsegmented_distribution = string_to_list("1: 15, 2: 53, 3: 275, 4: 354, 5: 176, 6: 27")
    flan_segmented_distribution = string_to_list("1: 18, 2: 37, 3: 203, 4: 367, 5: 228, 6: 45, 7: 1, 8: 1")
    opt_unsegmented_distribution = string_to_list("1: 23, 2: 53, 3: 259, 4: 363, 5: 178, 6: 24")
    opt_segmented_distribution = string_to_list("1: 36, 2: 36, 3: 234, 4: 333, 5: 209, 6: 51, 7: 1")
    target_distribution = string_to_list("1: 26, 2: 52, 3: 234, 4: 391, 5: 174, 6: 23")

    plot_answer_length_distribution(t5_unsegmented_distribution, t5_segmented_distribution, 
                                    flan_unsegmented_distribution, flan_segmented_distribution,
                                    opt_unsegmented_distribution, opt_segmented_distribution,
                                    target_distribution,
                                    '../figures/answer_length_distribution.pdf')
    
    # Legend
    plot_legend('../figures/legend.pdf')

    # Task 1
    task = 1

    recall_t5_unsegmented = [96.00, 94.00, 98.00, 98.00]
    recall_t5_segmented = [30.00, 50.00, 50.00, 54.00]
    recall_flan_t5_unsegmented = [98.00, 92.00, 96.00, 92.00]
    recall_flan_t5_segmented = [34.00, 42.00, 42.00, 44.00]
    recall_opt_unsegmented = [72.00, 86.00, 70.00, 84.00]
    recall_opt_segmented = [42.00, 42.00, 50.00, 44.00]

    answer_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_unsegmented = [100.00, 97.83, 100.00, 100.00]
    answer_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_segmented = [100.00, 100.00, 100.00, 100.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [96.00, 94.00, 98.00, 98.00]
    t5_segmented = [30.00, 50.00, 50.00, 54.00]
    flan_t5_unsegmented = [98.00, 90.00, 96.00, 92.00]
    flan_t5_segmented = [34.00, 42.00, 42.00, 44.00]
    opt_unsegmented = [72.00, 86.00, 70.00, 84.00]
    opt_segmented = [42.00, 42.00, 50.00, 44.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0.47, 0, 0]

    test_t5_unsegmented = [0.96, 1.92, 0.95, 0.00]
    test_t5_segmented = [25.45, 13.76, 14.81, 15.79]
    test_flan_t5_unsegmented = [0.95, 0.98, 1.90, 0.00]
    test_flan_t5_segmented = [26.72, 20.69, 17.59, 14.56]
    test_opt_unsegmented = [17.02, 6.80, 10.20, 2.04]
    test_opt_segmented = [26.67, 21.05, 22.58, 20.65]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 2
    task = 2

    recall_t5_unsegmented = [80.00, 90.00, 94.00, 90.00]
    recall_t5_segmented = [0.00, 2.00, 0.00, 2.00]
    recall_flan_t5_unsegmented = [90.00, 96.00, 82.00, 84.00]
    recall_flan_t5_segmented = [2.00, 0.00, 2.00, 0.00]
    recall_opt_unsegmented = [56.00, 68.00, 74.00, 68.00]
    recall_opt_segmented = [2.00, 0.00, 2.00, 2.00]

    answer_t5_unsegmented = [65.00, 71.11, 68.09, 53.33]
    answer_t5_segmented = [0.00, 100.00, 0.00, 100.00]
    answer_flan_t5_unsegmented = [57.78, 43.75, 75.61, 90.48]
    answer_flan_t5_segmented = [100.00, 0.00, 100.00, 0.00]
    answer_opt_unsegmented = [67.86, 55.88, 81.08, 79.41]
    answer_opt_segmented = [100.00, 0.00, 100.00, 100.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [52.00, 64.00, 64.00, 48.00]
    t5_segmented = [0.00, 2.00, 0.00, 2.00]
    flan_t5_unsegmented = [52.00, 42.00, 62.00, 76.00]
    flan_t5_segmented = [2.00, 0.00, 2.00, 0.00]
    opt_unsegmented = [38.00, 38.00, 60.00, 54.00]
    opt_segmented = [2.00, 0.00, 2.00, 2.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [2.48, 2.86, 1.44, 1.46]
    test_t5_segmented = [57.94, 51.20, 57.99, 51.87]
    test_flan_t5_unsegmented = [2.43, 0.49, 1.00, 1.97]
    test_flan_t5_segmented = [57.94, 58.26, 50.00, 45.45]
    test_opt_unsegmented = [10.11, 5.97, 7.00, 9.31]
    test_opt_segmented = [56.94, 64.09, 54.50, 47.12]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 3
    task = 3

    recall_t5_unsegmented = [92.00, 94.00, 98.00, 98.00]
    recall_t5_segmented = [6.00, 4.00, 8.00, 28.00]
    recall_flan_t5_unsegmented = [98.00, 98.00, 96.00, 98.00]
    recall_flan_t5_segmented = [2.00, 0.00, 6.00, 18.00]
    recall_opt_unsegmented = [84.00, 66.00, 80.00, 76.00]
    recall_opt_segmented = [2.00, 4.00, 12.00, 8.00]

    answer_t5_unsegmented = [76.09, 74.47, 79.59, 75.51]
    answer_t5_segmented = [100.00, 100.00, 50.00, 64.29]
    answer_flan_t5_unsegmented = [73.47, 85.71, 91.67, 81.63]
    answer_flan_t5_segmented = [0.00, 0.00, 100.00, 77.78]
    answer_opt_unsegmented = [83.33, 72.73, 85.00, 94.74]
    answer_opt_segmented = [100.00, 50.00, 100.00, 75.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [70.00, 70.00, 78.00, 74.00]
    t5_segmented = [6.00, 4.00, 4.00, 18.00]
    flan_t5_unsegmented = [72.00, 84.00, 88.00, 80.00]
    flan_t5_segmented = [0.00, 0.00, 6.00, 14.00]
    opt_unsegmented = [70.00, 48.00, 68.00, 72.00]
    opt_segmented = [2.00, 2.00, 12.00, 6.00]

    train_t5_unsegmented = [0.52, 0.52, 0.52, 0.52]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0.52, 0.52, 0.52, 0.52]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 1.04, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [2.06, 1.55, 0.52, 0.52]
    test_t5_segmented = [46.99, 54.04, 42.34, 21.30]
    test_flan_t5_unsegmented = [0.52, 0.52, 1.04, 0.52]
    test_flan_t5_segmented = [54.87, 56.00, 45.37, 28.96]
    test_opt_unsegmented = [4.57, 10.85, 4.06, 6.44]
    test_opt_segmented = [48.23, 53.33, 41.01, 43.53]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 4
    task = 4

    recall_t5_unsegmented = [82.00, 86.00, 96.00, 98.00]
    recall_t5_segmented = [24.00, 16.00, 16.00, 30.00]
    recall_flan_t5_unsegmented = [88.00, 76.00, 76.00, 94.00]
    recall_flan_t5_segmented = [18.00, 24.00, 20.00, 44.00]
    recall_opt_unsegmented = [78.00, 82.00, 86.00, 76.00]
    recall_opt_segmented = [20.00, 20.00, 30.00, 44.00]

    reasoning_t5_unsegmented = [92.68, 95.35, 93.75, 95.92]
    reasoning_t5_segmented = [91.67, 100.00, 100.00, 80.00]
    reasoning_flan_t5_unsegmented = [95.45, 100.00, 100.00, 97.87]
    reasoning_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    reasoning_opt_unsegmented = [94.87, 100.00, 97.67, 100.00]
    reasoning_opt_segmented = [100.00, 100.00, 100.00, 100.00]

    answer_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_segmented = [100.00, 100.00, 100.00, 100.00]

    plot_task_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [76.00, 82.00, 90.00, 94.00]
    t5_segmented = [22.00, 16.00, 16.00, 24.00]
    flan_t5_unsegmented = [84.00, 76.00, 76.00, 92.00]
    flan_t5_segmented = [18.00, 24.00, 20.00, 44.00]
    opt_unsegmented = [74.00, 82.00, 84.00, 76.00]
    opt_segmented = [20.00, 20.00, 30.00, 44.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [8.53, 8.46, 2.29, 0.76]
    test_t5_segmented = [55.28, 61.83, 66.15, 42.86]
    test_flan_t5_unsegmented = [5.47, 17.05, 16.28, 4.58]
    test_flan_t5_segmented = [54.55, 55.28, 60.00, 33.07]
    test_opt_unsegmented = [10.66, 13.08, 7.94, 17.74]
    test_opt_segmented = [55.81, 53.97, 46.55, 31.93]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 5
    task = 5

    recall_t5_unsegmented = [84.00, 76.00, 92.00, 92.00]
    recall_t5_segmented = [2.00, 2.00, 4.00, 12.00]
    recall_flan_t5_unsegmented = [86.00, 78.00, 86.00, 86.00]
    recall_flan_t5_segmented = [4.00, 2.00, 4.00, 18.00]
    recall_opt_unsegmented = [32.00, 56.00, 54.00, 68.00]
    recall_opt_segmented = [6.00, 6.00, 4.00, 4.00]

    reasoning_t5_unsegmented = [57.14, 52.63, 60.87, 47.83]
    reasoning_t5_segmented = [0.00, 0.00, 50.00, 33.33]
    reasoning_flan_t5_unsegmented = [69.77, 64.10, 46.51, 65.12]
    reasoning_flan_t5_segmented = [0.00, 0.00, 100.00, 55.56]
    reasoning_opt_unsegmented = [62.50, 42.86, 51.85, 50.00]
    reasoning_opt_segmented = [100.00, 100.00, 0.00, 100.00]

    answer_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_t5_segmented = [0.00, 0.00, 100.00, 100.00]
    answer_flan_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_segmented = [0.00, 0.00, 100.00, 100.00]
    answer_opt_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_segmented = [100.00, 100.00, 0.00, 100.00]

    plot_task_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [48.00, 40.00, 56.00, 44.00]
    t5_segmented = [0.00, 0.00, 2.00, 4.00]
    flan_t5_unsegmented = [60.00, 50.00, 40.00, 56.00]
    flan_t5_segmented = [0.00, 0.00, 4.00, 10.00]
    opt_unsegmented = [20.00, 24.00, 28.00, 34.00]
    opt_segmented = [6.00, 6.00, 0.00, 4.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [6.40, 6.83, 1.59, 2.01]
    test_t5_segmented = [60.57, 59.26, 68.38, 49.58]
    test_flan_t5_unsegmented = [6.00, 7.63, 4.90, 4.40]
    test_flan_t5_segmented = [59.83, 53.16, 43.80, 38.96]
    test_opt_unsegmented = [18.36, 7.11, 10.45, 7.69]
    test_opt_segmented = [50.00, 51.16, 45.54, 43.54]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 6
    task = 6

    recall_t5_unsegmented = [82.00, 90.00, 84.00, 96.00]
    recall_t5_segmented = [0.00, 0.00, 2.00, 2.00]
    recall_flan_t5_unsegmented = [82.00, 78.00, 82.00, 92.00]
    recall_flan_t5_segmented = [0.00, 2.00, 2.00, 4.00]
    recall_opt_unsegmented = [34.00, 42.00, 56.00, 72.00]
    recall_opt_segmented = [0.00, 2.00, 4.00, 2.00]

    answer_t5_unsegmented = [78.05, 91.11, 95.24, 85.42]
    answer_t5_segmented = [0.00, 0.00, 100.00, 100.00]
    answer_flan_t5_unsegmented = [75.61, 79.49, 90.24, 100.00]
    answer_flan_t5_segmented = [0.00, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [94.12, 76.19, 82.14, 91.67]
    answer_opt_segmented = [0.00, 100.00, 100.00, 100.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [64.00, 82.00, 80.00, 82.00]
    t5_segmented = [0.00, 0.00, 2.00, 2.00]
    flan_t5_unsegmented = [62.00, 62.00, 74.00, 92.00]
    flan_t5_segmented = [0.00, 2.00, 2.00, 4.00]
    opt_unsegmented = [32.00, 32.00, 46.00, 66.00]
    opt_segmented = [0.00, 2.00, 4.00, 2.00]

    train_t5_unsegmented = [0, 0.22, 0, 0]
    train_t5_segmented = [0, 0, 0, 0.22]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [8.44, 3.08, 6.19, 0.89]
    test_t5_segmented = [61.32, 61.41, 59.43, 55.42]
    test_flan_t5_unsegmented = [8.00, 7.52, 5.33, 1.79]
    test_flan_t5_segmented = [62.40, 61.22, 55.97, 40.00]
    test_opt_unsegmented = [33.18, 27.07, 21.24, 17.03]
    test_opt_segmented = [60.89, 58.92, 42.98, 48.28]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 7
    task = 7

    recall_t5_unsegmented = [78.00, 80.00, 80.00, 84.00]
    recall_t5_segmented = [6.00, 20.00, 24.00, 44.00]
    recall_flan_t5_unsegmented = [68.00, 82.00, 76.00, 92.00]
    recall_flan_t5_segmented = [14.00, 16.00, 26.00, 28.00]
    recall_opt_unsegmented = [56.00, 64.00, 64.00, 60.00]
    recall_opt_segmented = [14.00, 20.00, 22.00, 10.00]

    answer_t5_unsegmented = [64.10, 67.50, 65.00, 57.14]
    answer_t5_segmented = [66.67, 70.00, 75.00, 68.18]
    answer_flan_t5_unsegmented = [64.71, 90.24, 97.37, 97.83]
    answer_flan_t5_segmented = [57.14, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [32.14, 84.38, 81.25, 83.33]
    answer_opt_segmented = [57.14, 90.00, 81.82, 100.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [50.00, 54.00, 52.00, 48.00]
    t5_segmented = [4.00, 14.00, 18.00, 30.00]
    flan_t5_unsegmented = [44.00, 74.00, 74.00, 90.00]
    flan_t5_segmented = [8.00, 16.00, 26.00, 28.00]
    opt_unsegmented = [18.00, 54.00, 52.00, 50.00]
    opt_segmented = [8.00, 18.00, 18.00, 10.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0.28, 0, 0]

    test_t5_unsegmented = [12.43, 11.17, 7.22, 5.49]
    test_t5_segmented = [55.96, 44.97, 36.70, 25.52]
    test_flan_t5_unsegmented = [18.44, 11.24, 12.64, 3.95]
    test_flan_t5_segmented = [56.68, 50.00, 44.50, 33.85]
    test_opt_unsegmented = [24.73, 17.39, 20.00, 25.41]
    test_opt_segmented = [61.22, 53.16, 45.73, 54.12]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 8
    task = 8

    recall_t5_unsegmented = [98.00, 100.00, 88.00, 92.00]
    recall_t5_segmented = [4.00, 12.00, 10.00, 22.00]
    recall_flan_t5_unsegmented = [80.00, 88.00, 74.00, 80.00]
    recall_flan_t5_segmented = [8.00, 8.00, 6.00, 10.00]
    recall_opt_unsegmented = [68.00, 76.00, 82.00, 80.00]
    recall_opt_segmented = [10.00, 2.00, 12.00, 16.00]

    reasoning_t5_unsegmented = [85.71, 96.00, 100.00, 76.09]
    reasoning_t5_segmented = [100.00, 100.00, 100.00, 72.73]
    reasoning_flan_t5_unsegmented = [97.50, 97.73, 97.30, 95.00]
    reasoning_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    reasoning_opt_unsegmented = [97.06, 94.74, 100, 100.00]
    reasoning_opt_segmented = [100.00, 100.00, 100.00, 100.00]

    answer_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_segmented = [100.00, 100.00, 100.00, 100.00]

    plot_task_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [84.00, 96.00, 88.00, 70.00]
    t5_segmented = [4.00, 12.00, 10.00, 16.00]
    flan_t5_unsegmented = [78.00, 86.00, 72.00, 76.00]
    flan_t5_segmented = [8.00, 8.00, 6.00, 10.00]
    opt_unsegmented = [66.00, 72.00, 82.00, 80.00]
    opt_segmented = [10.00, 2.00, 12.00, 16.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [2.26, 0.00, 1.17, 0.00]
    test_t5_segmented = [65.32, 57.14, 58.92, 36.11]
    test_flan_t5_unsegmented = [11.80, 6.29, 13.64, 9.83]
    test_flan_t5_segmented = [60.92, 57.93, 52.41, 48.02]
    test_opt_unsegmented = [14.79, 9.25, 9.04, 9.88]
    test_opt_segmented = [62.36, 63.24, 54.02, 52.81]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 9
    task = 9

    recall_t5_unsegmented = [100.00, 100.00, 100.00, 98.00]
    recall_t5_segmented = [4.00, 4.00, 6.00, 14.00]
    recall_flan_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    recall_flan_t5_segmented = [10.00, 16.00, 8.00, 10.00]
    recall_opt_unsegmented = [56.00, 86.00, 68.00, 78.00]
    recall_opt_segmented = [14.00, 8.00, 12.00, 6.00]

    reasoning_t5_unsegmented = [74.00, 86.00, 80.00, 63.27]
    reasoning_t5_segmented = [100.00, 100.00, 100.00, 85.71]
    reasoning_flan_t5_unsegmented = [80.00, 70.00, 78.00, 72.00]
    reasoning_flan_t5_segmented = [100.00, 87.50, 100.00, 60.00]
    reasoning_opt_unsegmented = [60.71, 67.44, 73.53, 84.62]
    reasoning_opt_segmented = [85.71, 100.00, 83.33, 66.67]

    answer_t5_unsegmented = [24.32, 34.88, 25.00, 25.81]
    answer_t5_segmented = [0.00, 0.00, 33.33, 16.67]
    answer_flan_t5_unsegmented = [20.00, 34.29, 28.21, 30.56]
    answer_flan_t5_segmented = [0.00, 0.00, 25.00, 0.00]
    answer_opt_unsegmented = [35.29, 20.69, 40.00, 45.45]
    answer_opt_segmented = [33.33, 25.00, 40.00, 0.00]

    plot_task_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [18.00, 30.00, 20.00, 16.00]
    t5_segmented = [0.00, 0.00, 2.00, 2.00]
    flan_t5_unsegmented = [16.00, 24.00, 22.00, 22.00]
    flan_t5_segmented = [0.00, 0.00, 2.00, 0.00]
    opt_unsegmented = [12.00, 12.00, 20.00, 30.00]
    opt_segmented = [4.00, 2.00, 4.00, 0.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0.28, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [0.00, 0.00, 0.00, 1.14]
    test_t5_segmented = [50.57, 50.60, 50.30, 41.72]
    test_flan_t5_unsegmented = [0.00, 0.00, 0.00, 0.00]
    test_flan_t5_segmented = [51.52, 45.09, 47.70, 48.63]
    test_opt_unsegmented = [16.57, 4.57, 9.71, 6.29]
    test_opt_segmented = [47.65, 50.00, 46.39, 45.03]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 10
    task = 10

    recall_t5_unsegmented = [82.00, 96.00, 92.00, 100.00]
    recall_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    recall_flan_t5_unsegmented = [82.00, 96.00, 84.00, 94.00]
    recall_flan_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    recall_opt_unsegmented = [66.00, 82.00, 82.00, 86.00]
    recall_opt_segmented = [0.00, 0.00, 0.00, 0.00]

    answer_t5_unsegmented = [39.02, 41.67, 26.09, 24.00]
    answer_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    answer_flan_t5_unsegmented = [53.66, 45.83, 78.57, 65.96]
    answer_flan_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    answer_opt_unsegmented = [24.24, 31.71, 53.66, 62.79]
    answer_opt_segmented = [0.00, 0.00, 0.00, 0.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [32.00, 40.00, 24.00, 24.00]
    t5_segmented = [0.00, 0.00, 0.00, 0.00]
    flan_t5_unsegmented = [44.00, 44.00, 66.00, 62.00]
    flan_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    opt_unsegmented = [16.00, 26.00, 44.00, 54.00]
    opt_segmented = [0.00, 0.00, 0.00, 0.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0.22, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [10.13, 2.67, 5.73, 0.00]
    test_t5_segmented = [89.45, 82.55, 83.87, 71.67]
    test_flan_t5_unsegmented = [9.25, 2.67, 7.11, 3.12]
    test_flan_t5_segmented = [91.30, 88.93, 82.83, 84.40]
    test_opt_unsegmented = [21.08, 7.62, 7.21, 11.89]
    test_opt_segmented = [96.09, 93.56, 92.62, 85.23]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 11
    task = 11

    recall_t5_unsegmented = [86.00, 90.00, 86.00, 92.00]
    recall_t5_segmented = [0.00, 12.00, 12.00, 38.00]
    recall_flan_t5_unsegmented = [90.00, 96.00, 94.00, 100.00]
    recall_flan_t5_segmented = [4.00, 10.00, 4.00, 18.00]
    recall_opt_unsegmented = [44.00, 72.00, 90.00, 78.00]
    recall_opt_segmented = [0.00, 6.00, 8.00, 14.00]

    answer_t5_unsegmented = [53.49, 77.78, 69.77, 65.22]
    answer_t5_segmented = [0.00, 50.00, 100.00, 63.16]
    answer_flan_t5_unsegmented = [57.78, 68.75, 97.87, 88.00]
    answer_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [68.18, 55.56, 66.67, 74.36]
    answer_opt_segmented = [0.00, 66.67, 100.00, 100.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [46.00, 70.00, 60.00, 60.00]
    t5_segmented = [0.00, 6.00, 12.00, 24.00]
    flan_t5_unsegmented = [52.00, 66.00, 92.00, 88.00]
    flan_t5_segmented = [4.00, 10.00, 4.00, 18.00]
    opt_unsegmented = [30.00, 40.00, 60.00, 58.00]
    opt_segmented = [0.00, 4.00, 8.00, 14.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [9.36, 5.78, 6.86, 2.91]
    test_t5_segmented = [63.68, 49.45, 41.67, 17.89]
    test_flan_t5_unsegmented = [5.17, 2.35, 2.92, 0.00]
    test_flan_t5_segmented = [57.84, 50.79, 47.87, 32.46]
    test_opt_unsegmented = [26.14, 12.50, 6.90, 9.83]
    test_opt_segmented = [52.36, 56.99, 46.35, 45.55]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 12
    task = 12

    recall_t5_unsegmented = [74.00, 78.00, 78.00, 74.00]
    recall_t5_segmented = [14.00, 30.00, 24.00, 32.00]
    recall_flan_t5_unsegmented = [68.00, 82.00, 62.00, 70.00]
    recall_flan_t5_segmented = [12.00, 24.00, 26.00, 32.00]
    recall_opt_unsegmented = [62.00, 68.00, 76.00, 76.00]
    recall_opt_segmented = [14.00, 20.00, 28.00, 22.00]

    answer_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_segmented = [100.00, 100.00, 100.00, 100.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [74.00, 78.00, 78.00, 74.00]
    t5_segmented = [14.00, 30.00, 24.00, 32.00]
    flan_t5_unsegmented = [68.00, 82.00, 62.00, 70.00]
    flan_t5_segmented = [12.00, 24.00, 26.00, 32.00]
    opt_unsegmented = [62.00, 68.00, 76.00, 76.00]
    opt_segmented = [14.00, 20.00, 28.00, 22.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [11.36, 10.23, 10.67, 9.66]
    test_t5_segmented = [33.70, 30.41, 29.24, 27.06]
    test_flan_t5_unsegmented = [14.45, 9.09, 14.94, 12.50]
    test_flan_t5_segmented = [41.67, 34.29, 32.76, 28.00]
    test_opt_unsegmented = [18.75, 11.60, 9.83, 10.00]
    test_opt_segmented = [40.11, 40.74, 35.68, 34.86]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 13
    task = 13

    recall_t5_unsegmented = [96.00, 76.00, 80.00, 88.00]
    recall_t5_segmented = [8.00, 8.00, 4.00, 2.00]
    recall_flan_t5_unsegmented = [80.00, 88.00, 82.00, 88.00]
    recall_flan_t5_segmented = [4.00, 8.00, 4.00, 16.00]
    recall_opt_unsegmented = [52.00, 64.00, 80.00, 74.00]
    recall_opt_segmented = [4.00, 2.00, 4.00, 14.00]

    reasoning_t5_unsegmented = [47.92, 31.58, 37.50, 22.73]
    reasoning_t5_segmented = [100.00, 75.00, 100.00, 100.00]
    reasoning_flan_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    reasoning_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    reasoning_opt_unsegmented = [100.00, 100.00, 97.5, 100.00]
    reasoning_opt_segmented = [100.00, 100.00, 100.00, 100.00]

    answer_t5_unsegmented = [65.22, 75.00, 93.33, 70.00]
    answer_t5_segmented = [75.00, 100.00, 100.00, 100.00]
    answer_flan_t5_unsegmented = [100.00, 100.00, 100.00, 95.45]
    answer_flan_t5_segmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [88.46, 100.00, 94.87, 100.00]
    answer_opt_segmented = [100.00, 100.00, 100.00, 100.00]

    plot_task_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [30.00, 18.00, 28.00, 14.00]
    t5_segmented = [6.00, 6.00, 4.00, 2.00]
    flan_t5_unsegmented = [80.00, 88.00, 82.00, 84.00]
    flan_t5_segmented = [4.00, 8.00, 4.00, 16.00]
    opt_unsegmented = [46.00, 64.00, 74.00, 74.00]
    opt_segmented = [4.00, 2.00, 4.00, 14.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [0.00, 1.73, 2.82, 1.68]
    test_t5_segmented = [57.14, 54.88, 67.84, 60.12]
    test_flan_t5_unsegmented = [5.11, 5.08, 4.44, 1.68]
    test_flan_t5_segmented = [69.64, 59.51, 63.03, 48.84]
    test_opt_unsegmented = [19.41, 5.36, 3.93, 3.51]
    test_opt_segmented = [54.04, 65.85, 55.97, 42.04]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 14
    task = 14

    recall_t5_unsegmented = [90.00, 94.00, 92.00, 94.00]
    recall_t5_segmented = [2.00, 6.00, 0.00, 6.00]
    recall_flan_t5_unsegmented = [86.00, 88.00, 94.00, 92.00]
    recall_flan_t5_segmented = [0.00, 2.00, 2.00, 2.00]
    recall_opt_unsegmented = [60.00, 72.00, 64.00, 74.00]
    recall_opt_segmented = [2.00, 2.00, 4.00, 10.00]

    answer_t5_unsegmented = [57.78, 89.36, 95.65, 82.98]
    answer_t5_segmented = [0.00, 33.33, 0.00, 100.00]
    answer_flan_t5_unsegmented = [97.67, 100.00, 100.00, 93.48]
    answer_flan_t5_segmented = [0.00, 100.00, 100.00, 100.00]
    answer_opt_unsegmented = [100.00, 97.22, 81.25, 97.30]
    answer_opt_segmented = [100.00, 100.00, 100.00, 80.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [52.00, 84.00, 88.00, 78.00]
    t5_segmented = [0.00, 2.00, 0.00, 6.00]
    flan_t5_unsegmented = [84.00, 88.00, 94.00, 86.00]
    flan_t5_segmented = [0.00, 2.00, 2.00, 2.00]
    opt_unsegmented = [60.00, 70.00, 52.00, 72.00]
    opt_segmented = [2.00, 2.00, 4.00, 8.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 1.01, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [6.62, 2.67, 5.92, 4.55]
    test_t5_segmented = [64.53, 64.84, 67.98, 61.88]
    test_flan_t5_unsegmented = [8.55, 6.62, 6.00, 2.65]
    test_flan_t5_segmented = [74.14, 63.91, 59.54, 59.20]
    test_opt_unsegmented = [17.39, 6.80, 7.14, 8.22]
    test_opt_segmented = [65.41, 66.88, 57.93, 54.07]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 15
    task = 15

    recall_t5_unsegmented = [96.00, 96.00, 94.00, 98.00]
    recall_t5_segmented = [4.00, 2.00, 4.00, 10.00]
    recall_flan_t5_unsegmented = [98.00, 98.00, 98.00, 98.00]
    recall_flan_t5_segmented = [0.00, 6.00, 0.00, 2.00]
    recall_opt_unsegmented = [50.00, 78.00, 78.00, 90.00]
    recall_opt_segmented = [2.00, 0.00, 6.00, 8.00]

    answer_t5_unsegmented = [75.00, 81.25, 78.72, 73.47]
    answer_t5_segmented = [50.00, 0.00, 50.00, 40.00]
    answer_flan_t5_unsegmented = [73.47, 73.47, 83.67, 83.67]
    answer_flan_t5_segmented = [0.00, 66.67, 0.00, 100.00]
    answer_opt_unsegmented = [64.00, 74.36, 79.49, 75.56]
    answer_opt_segmented = [100.00, 0.00, 33.33, 100.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [72.00, 78.00, 74.00, 72.00]
    t5_segmented = [2.00, 0.00, 2.00, 4.00]
    flan_t5_unsegmented = [72.00, 72.00, 82.00, 82.00]
    flan_t5_segmented = [0.00, 4.00, 0.00, 2.00]
    opt_unsegmented = [32.00, 58.00, 62.00, 68.00]
    opt_segmented = [2.00, 0.00, 2.00, 8.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [0.78, 0.78, 1.18, 0.39]
    test_t5_segmented = [44.69, 47.62, 43.40, 39.72]
    test_flan_t5_unsegmented = [0.39, 0.39, 0.39, 0.39]
    test_flan_t5_segmented = [47.62, 44.36, 45.52, 42.70]
    test_opt_unsegmented = [15.00, 3.53, 9.88, 2.00]
    test_opt_segmented = [43.90, 45.32, 41.70, 40.60]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 16
    task = 16

    recall_t5_unsegmented = [96.00, 100.00, 94.00, 98.00]
    recall_t5_segmented = [2.00, 4.00, 8.00, 32.00]
    recall_flan_t5_unsegmented = [100.00, 98.00, 100.00, 98.00]
    recall_flan_t5_segmented = [4.00, 2.00, 2.00, 12.00]
    recall_opt_unsegmented = [82.00, 94.00, 84.00, 96.00]
    recall_opt_segmented = [0.00, 4.00, 4.00, 0.00]

    answer_t5_unsegmented = [33.33, 60.00, 93.62, 93.88]
    answer_t5_segmented = [0.00, 0.00, 100.00, 93.75]
    answer_flan_t5_unsegmented = [50.00, 51.02, 58.00, 79.59]
    answer_flan_t5_segmented = [50.00, 0.00, 100.00, 83.33]
    answer_opt_unsegmented = [60.98, 51.06, 33.33, 75.00]
    answer_opt_segmented = [0.00, 0.00, 100.00, 0.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [32.00, 60.00, 88.00, 92.00]
    t5_segmented = [0.00, 0.00, 8.00, 30.00]
    flan_t5_unsegmented = [50.00, 50.00, 58.00, 78.00]
    flan_t5_segmented = [2.00, 0.00, 2.00, 10.00]
    opt_unsegmented = [50.00, 48.00, 28.00, 72.00]
    opt_segmented = [0.00, 0.00, 4.00, 0.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0.89, 0]
    train_opt_segmented = [0, 0.22, 0, 0]

    test_t5_unsegmented = [4.02, 0.00, 1.36, 0.45]
    test_t5_segmented = [59.07, 52.65, 44.44, 17.67]
    test_flan_t5_unsegmented = [0.00, 0.45, 0.00, 1.35]
    test_flan_t5_segmented = [61.07, 59.50, 51.00, 36.48]
    test_opt_unsegmented = [9.38, 2.67, 6.19, 0.89]
    test_opt_segmented = [56.38, 54.66, 44.03, 52.28]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 17
    task = 17

    recall_t5_unsegmented = [92.00, 86.00, 88.00, 96.00]
    recall_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    recall_flan_t5_unsegmented = [96.00, 94.00, 100.00, 94.00]
    recall_flan_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    recall_opt_unsegmented = [72.00, 82.00, 76.00, 84.00]
    recall_opt_segmented = [0.00, 0.00, 0.00, 2.00]

    answer_t5_unsegmented = [100.00, 100.00, 100.00, 70.83]
    answer_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    answer_flan_t5_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_flan_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    answer_opt_unsegmented = [100.00, 100.00, 100.00, 100.00]
    answer_opt_segmented = [0.00, 0.00, 0.00, 100.00]

    plot_task_accuracy_breakdown_no_reasoning(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [92.00, 86.00, 88.00, 68.00]
    t5_segmented = [0.00, 0.00, 0.00, 0.00]
    flan_t5_unsegmented = [96.00, 94.00, 100.00, 94.00]
    flan_t5_segmented = [0.00, 0.00, 0.00, 0.00]
    opt_unsegmented = [72.00, 82.00, 76.00, 84.00]
    opt_segmented = [0.00, 0.00, 0.00, 2.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [4.98, 11.42, 7.34, 2.26]
    test_t5_segmented = [72.22, 70.35, 67.09, 66.53]
    test_flan_t5_unsegmented = [3.64, 4.11, 0.00, 1.36]
    test_flan_t5_segmented = [83.63, 74.35, 67.26, 69.23]
    test_opt_unsegmented = [14.29, 7.21, 9.57, 4.17]
    test_opt_segmented = [80.80, 80.69, 67.51, 67.09]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')

    # Task 18
    task = 18

    recall_t5_unsegmented = [84.00, 90.00, 94.00, 92.00]
    recall_t5_segmented = [24.00, 36.00, 42.00, 46.00]
    recall_flan_t5_unsegmented = [70.00, 90.00, 98.00, 94.00]
    recall_flan_t5_segmented = [12.00, 18.00, 16.00, 44.00]
    recall_opt_unsegmented = [94.00, 88.00, 84.00, 90.00]
    recall_opt_segmented = [16.00, 16.00, 46.00, 44.00]

    reasoning_t5_unsegmented = [78.57, 86.67, 80.85, 71.74]
    reasoning_t5_segmented = [75.00, 88.89, 85.71, 73.91]
    reasoning_flan_t5_unsegmented = [74.29, 73.33, 81.63, 80.85]
    reasoning_flan_t5_segmented = [83.33, 77.78, 75.00, 68.18]
    reasoning_opt_unsegmented = [82.98, 75.00, 76.19, 84.44]
    reasoning_opt_segmented = [87.50, 62.50, 86.96, 81.82]

    answer_t5_unsegmented = [84.85, 92.31, 92.11, 90.91]
    answer_t5_segmented = [88.89, 100.00, 94.44, 94.12]
    answer_flan_t5_unsegmented = [96.15, 93.94, 95.00, 94.74]
    answer_flan_t5_segmented = [100.00, 100.00, 100.00, 93.33]
    answer_opt_unsegmented = [94.87, 87.88, 90.62, 97.37]
    answer_opt_segmented = [71.43, 100.00, 95.00, 100.00]

    plot_task_accuracy_breakdown(recall_t5_unsegmented, recall_t5_segmented, recall_flan_t5_unsegmented, recall_flan_t5_segmented, recall_opt_unsegmented, recall_opt_segmented, 
                            reasoning_t5_unsegmented, reasoning_t5_segmented, reasoning_flan_t5_unsegmented, reasoning_flan_t5_segmented, reasoning_opt_unsegmented, reasoning_opt_segmented, 
                            answer_t5_unsegmented, answer_t5_segmented, answer_flan_t5_unsegmented, answer_flan_t5_segmented, answer_opt_unsegmented, answer_opt_segmented, 
                            t5_sizes, opt_sizes,
                            f'../figures/task{task}_accuracy_breakdown.pdf')

    t5_unsegmented = [56.00, 72.00, 70.00, 60.00]
    t5_segmented = [16.00, 32.00, 34.00, 32.00]
    flan_t5_unsegmented = [50.00, 62.00, 76.00, 72.00]
    flan_t5_segmented = [10.00, 14.00, 12.00, 28.00]
    opt_unsegmented = [74.00, 58.00, 58.00, 74.00]
    opt_segmented = [10.00, 10.00, 38.00, 36.00]

    train_t5_unsegmented = [0, 0, 0, 0]
    train_t5_segmented = [0, 0, 0, 0]
    train_flan_t5_unsegmented = [0, 0, 0, 0]
    train_flan_t5_segmented = [0, 0, 0, 0]
    train_opt_unsegmented = [0, 0, 0, 0]
    train_opt_segmented = [0, 0, 0, 0]

    test_t5_unsegmented = [4.52, 2.78, 0.58, 1.69]
    test_t5_segmented = [38.83, 27.23, 25.13, 21.99]
    test_flan_t5_unsegmented = [11.24, 3.39, 0.00, 2.25]
    test_flan_t5_segmented = [46.32, 41.49, 39.04, 24.74]
    test_opt_unsegmented = [1.68, 3.28, 7.22, 2.75]
    test_opt_segmented = [33.85, 39.68, 23.94, 21.16]

    plot_task_whole_accuracy_and_hallucination_rate(t5_unsegmented, t5_segmented, flan_t5_unsegmented, flan_t5_segmented, opt_unsegmented, opt_segmented,
                                                    train_t5_unsegmented, train_t5_segmented, train_flan_t5_unsegmented, train_flan_t5_segmented, train_opt_unsegmented, train_opt_segmented,
                                                    test_t5_unsegmented, test_t5_segmented, test_flan_t5_unsegmented, test_flan_t5_segmented, test_opt_unsegmented, test_opt_segmented,
                                                    t5_sizes, opt_sizes,
                                                    f'../figures/task{task}_whole_answer_and_hallucination_rate.pdf')


if __name__ == '__main__':
    main()