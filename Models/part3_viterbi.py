import sys
from collections import defaultdict
import time
from itertools import groupby

# Local File Import
import part2_simple_system as EM

# ===================================================================================================================
# Write a function that estimates the transition parameters from training set using MLE (Maximum Likelihood Estimation)
# q(Yi|Yi-1) = [ Count(Yi-1, Yi) / Count(Yi-1) ]
# ___________________________________________________________________________________________________________________
def get_transition_params(file):
    # Dictionaries to store Tag and Word counts
    tag_count_dict = defaultdict(int)
    tag_word_count_dict = defaultdict(lambda: defaultdict(int))

    lines = file.readlines()  # We cannot strip here.. Need to know where is empty line to signal end of sentence

    """
    # Consider Transition from START(Y0) to First Tag(Y1)
    first_tag = lines[0].strip().split()[1]  # first tag in first line - tuple of (word,tag)
    tag_word_count_dict[first_tag]["START"] += 1
    tag_count_dict[first_tag] += 1
    tag_count_dict["START"] += 1
    """

    # Add empty line to start of lines list
    # Important to signal start of sentence for the first sentence in input file
    lines.insert(0, "\n")

    for i in range(1, len(lines)):
        # Note line can be empty here. Empty lines separates sentences - Adjacent lines are Beginning & End of Sentences
        line = lines[i]
        prev_line = lines[i-1]

        # CASE 1: Beginning of Sentence
        if prev_line.rstrip() == "":
            tag_i = line.strip().split()[1]

            # Transition to Tag(i) from START
            tag_word_count_dict[tag_i]["START"] += 1
            tag_count_dict[tag_i] += 1
            tag_count_dict["START"] += 1

        # CASE 2: End of Sentence
        if line.rstrip() == "":
            tag_iprev = prev_line.strip().split()[1]  # tag_iprev is Tag(i-1) at (i-1)th position

            # Transition to STOP from Tag(i-1)
            tag_word_count_dict["STOP"][tag_iprev] += 1
            tag_count_dict["STOP"] += 1

        # CASE 3: Inside Sentence (not Beginning or End)
        if prev_line.rstrip() != "" and line.rstrip() != "":
            tag_i = line.strip().split()[1]
            tag_iprev = prev_line.strip().split()[1]

            # Transtion to STOP from Tag(i-1)
            tag_word_count_dict[tag_i][tag_iprev] += 1
            tag_count_dict[tag_i] += 1

        # Transition Parameters - q(Yi/Yi-1)
        transition_params = {}
        for tag_i in tag_count_dict.keys():
            transition_params[tag_i] = {}
            for tag_iprev in tag_word_count_dict[tag_i].keys():
                transition_params[tag_i][tag_iprev] = float(tag_word_count_dict[tag_i][tag_iprev]/tag_count_dict[tag_iprev])

    return transition_params


# ===================================================================================================================
# VITERBI ALGORITHM
# Use the estimated transition and emission parameters to implement the Viterbi Algorithm to compute the following
# Y1*,...,Yn* = arg-max-Y1,...,Yn P(X1,...,Xn, Y1,...,Yn)
# ___________________________________________________________________________________________________________________
def inner_viterbi(sentence, _emission_params, _transition_params, _tag_count_dict):
    return 0


def formatted_x_seqs_list(_lines):
    # Returns a List of X Sequences (or sentences) in the form of:
    # e.g. [ ["begin", "mid", "end"], [], ["begin", "mid", "end"], []]
    # Note: empty list [] represent empty lines
    total_rows = len(_lines)
    result_list = []
    x_seq = []
    for i in range(total_rows):
        word = _lines[i]
        if word != "\n":
            x_seq.append(word)
            if i == (total_rows-1):
                result_list.append(x_seq)
        else:
            if len(x_seq) > 0:
                result_list.append(x_seq)
            result_list.append([])
            x_seq = []
    return result_list

def main_viterbi(_x_seq, _emission_params, _transition_params, _tag_count_dict):
    # ___________________________________________________
    # Forward Algorithm - Left to Right (START to STOP)
    # ___________________________________________________


    # ___________________________________________________
    # Backward Algorithm - Right to Left (STOP to START)
    # ___________________________________________________
    return 0


def viterbi(_input_file, _emission_params, _transition_params, _tag_count_dict):
    lines = _input_file.readlines()
    x_seqs_list = formatted_x_seqs_list(lines)  # List of X Sequences (or sentences)

    y_seqs_list = [] # Store the predicted y sequences in a list (including empty lines)
    for x_seq in x_seqs_list:
        if len(x_seq) == 0:  # If empty list, means it row is supposed to be an empty line
            y_seqs_list.append(["\n"])
        else:  # Non-empty list - sentence
            # List of predicted y sequence for a given x sequence
            y_seq = main_viterbi(x_seq, _emission_params, _transition_params, _tag_count_dict)
            y_seqs_list.append(y_seq)
    return y_seqs_list



if __name__ == "__main__":
    k = 0.5
    start_time = time.time()
    file_path = sys.argv[1]

    # Get calculated Emission Parameters
    with open("{}/train".format(file_path), "r") as training_file:
        emission_params, train_tag_count_dict, train_word_count_dict = EM.get_fixed_emission_params(training_file, 0.5)

    # Get calculated Transition Parameters
    with open("{}/train".format(file_path), "r") as training_file:
        transition_params = get_transition_params(training_file)

    # Get Predicted Y Sequences from Viterbi Algorithm
    with open("{}/dev.in".format(file_path), "r") as input_file:
        predicted_Y_seqs = viterbi(input_file, emission_params, transition_params, train_tag_count_dict)

    # Write Predicted Y Sequences to output file
    # with open("{}/dev.p3.out".format(file_path), "w+") as output_file:
    #     generate_predictions(output_file, input_file, predicted_Y_seqs)


    # Predict and write prediction results
    # with open("{}/dev.p2.out".format(file_path), "w+") as results_file:
    #     with open("{}/dev.in".format(file_path), "r") as input_file:
    #         prediction(fixed_emission_params, train_tag_count_dict, train_word_count_dict, input_file, results_file)

    stop_time = time.time()
    print("Time taken:", stop_time - start_time)