import sys
from collections import defaultdict
import time
from itertools import groupby
from math import log
from sys import maxsize  # Arbitrary large value

# Local File Import
import part2_simple_system as EM

# ===================================================================================================================
# Write a function that estimates the transition parameters from training set using MLE (Maximum Likelihood Estimation)
# q(Yi|Yi-1) = [ Count(Yi-1, Yi) / Count(Yi-1) ]
# ___________________________________________________________________________________________________________________
def get_transition_params(file):
    """
    :param file: [file] file with word and tag in each line
    :return: [dict] Store transition parameters in the form of dictionary[Yi][Yi-1] where Yi=current_tag, Yi-1=previous_tag
    """
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
""" # Not in use anymore
# Helper function to parse file and return X sequences
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
"""

def viterbi(_x_seq, _emission_params, _transition_params, _tags_list):
    """
    :param _x_seq: [list] List of X sequence (or a sentence)
    :param _emission_params:  [dict] emission parameters
    :param _transition_params: [dict] transition parameters
    :param _tags_list: [list] 2D List of tags in transition parameters (without START and STOP tags)
    :return: [list] Predicted Y sequences in a flat 1D list
             [list] Pi List
             [list] Args list
    """
    # Initialise some common variables
    n = len(_x_seq)  # Number of words in _x_seq
    pi_list = [[0 for tag in range(len(_tags_list))] for j in range(n)]  # List of current max scores e.g. [ ["A", "B", "C"], ["X "Y","Z"] ]
    args_list = [[0 for arg in range(len(_tags_list))] for j in range(n)]  # Argmax List

    # ___________________________________________________
    # Forward Algorithm - Left to Right (From START)
    # ___________________________________________________

    # 1. Initialisation - START to first word
    for i in range(len(_tags_list)):
        word = _x_seq[0].strip()


        # Dynamically handle the different cases,e.g. if no "START" in dict
        try:
            tr = _transition_params[_tags_list[i]]["START"]
            #em = EM.get_emission(_emission_params, _x_seq, _tags_list[i], word)  # def get_emission(e_params, _train_word_count_dict, tag, word): train_word_count_dict
            em = EM.get_emission(_emission_params, train_word_count_dict, _tags_list[i], word)  # def get_emission(e_params, _train_word_count_dict, tag, word):

            if em == 0:
                em_score = -maxsize  # Set large constant negative value
            else:
                em_score = log(em)  # Use logarithmic function
            tr_score = log(tr)
            pi_list[0][i] = tr_score + em_score
        except KeyError:
            if em == 0:
                em_score = -maxsize
            else:
                em_score = log(em)
            pi_list[0][i] = -maxsize + em_score
        except TypeError:
            pi_list[0][i] = -maxsize


    # 2. Main Loop
    for i in range(1, n):  # k -> i
        word = _x_seq[i].strip()
        for j in range(len(_tags_list)):  # v -> j
            max_list = []  # Store the max score values
            arg_list = []  # Store the arg values

            for k in range(len(_tags_list)):  # u -> k

                # Dynamically handle the different cases,e.g. if no particular tag in dict
                try:
                    prev_pi = pi_list[i-1][k]

                    tr = _transition_params[_tags_list[j]][_tags_list[k]]
                    # em = EM.get_emission(_emission_params, _x_seq, _tags_list[j], word)
                    em = EM.get_emission(_emission_params, train_word_count_dict, _tags_list[j], word)  # def get_emission(e_params, _train_word_count_dict, tag, word):

                    if em == 0:
                        em_score = -maxsize  # Set large constant negative value
                    else:
                        em_score = log(em)  # Use logarithmic function

                    curr_val = prev_pi + log(tr) + em_score
                    curr_arg_val = prev_pi + log(tr)

                except KeyError:
                    if em == 0:
                        em_score = -maxsize  # Set large constant negative value
                    else:
                        em_score = log(em)  # Use logarithmic function
                    curr_val = prev_pi - maxsize + em_score
                    curr_arg_val = prev_pi - maxsize

                max_list.append(curr_val)
                arg_list.append(curr_arg_val)

            pi_list[i][j] = max(max_list)
            args_list[i][j] = arg_list.index(max(arg_list)) + 1

    # ___________________________________________________
    # Backward Algorithm - Right to Left (From STOP)
    # ___________________________________________________
    y_seq = [0]*n  # To store predicted best Y sequences
    idx_list = [0]*n  # To store index positions

    # 1. Start Backtracking from STOP to last tag
    last_tag_layer = pi_list[n-1]
    final_check = []

    # Get max val
    for i in range(len(last_tag_layer)):
        prev_pi = last_tag_layer[i]

        # Dynamically handle the different cases
        try:
            tr = _transition_params["STOP"][_tags_list[i]]
            curr_val = prev_pi + log(tr)
        except KeyError:
            curr_val = prev_pi - maxsize
        final_check.append(curr_val)

    # Get last tag
    curr_max = -maxsize
    for j in range(len(final_check)):
        curr_val = final_check[j]
        if curr_val > curr_max:
            curr_max = curr_val
    if curr_max == -maxsize:
        y_seq[n-1] = tags_list[idx_list[n-1]]
        idx_list[n-1] = unknown_idx
    else:
        idx_list[n-1] = final_check.index(curr_max) + 1
        y_seq[n-1] = tags_list[final_check.index(curr_max)]

    # 2. Main Backtracking Loop - use stored arg
    for k in range(n-1, 0, -1):  # Loop from the back - R to L
        idx_list[k-1] = args_list[k][idx_list[k]-1]
        y_seq[k-1] = tags_list[idx_list[k-1]-1]

    return y_seq, pi_list, args_list


def get_viterbi_predictions(_input_file, _emission_params, _transition_params, _tags_list):
    """
    :param _input_file: [file] input txt file with word and tag
    :param _emission_params: [dict] emission parameters
    :param _transition_params: [dict] transition parameters
    :param _tags_list: [list] 2D List of tags in transition parameters (without START and STOP tags)
    :return: [list] Predicted Y sequences in a flat 1D list
    """
    lines = _input_file.readlines()
    # x_seqs_list = formatted_x_seqs_list(lines)  # List of X Sequences (or sentences)
    x_seqs_list = [list(group) for k, group in groupby(lines, lambda x: x == "\n") if not k]
    y_seqs_list = []  # Store the predicted y sequences in a list (including empty lines)
    """
    for x_seq in x_seqs_list:
        if len(x_seq) == 0:  # If empty list, means it row is supposed to be an empty line
            y_seqs_list.append(["\n"])
        else:  # Non-empty list - sentence
            # List of predicted y sequence for a given x sequence
            y_seq, pi_list, args_list = viterbi(x_seq, _emission_params, _transition_params, _tags_list)
            y_seqs_list.append(y_seq)
    """
    for i in range(len(x_seqs_list)):
        y_seq, pi_list, args_list = viterbi(x_seqs_list[i], _emission_params, _transition_params, _tags_list)
        y_seqs_list += y_seq
        y_seqs_list += [0]  # 0 as a separator between sentences
    return y_seqs_list

# ===================================================================================================================
# Generate predictions
# ___________________________________________________________________________________________________________________
def generate_predictions(_predicted_Y_seqs, _input_file, _output_file):
    """
    :param _predicted_Y_seqs: [list] Predicted Y sequences in a flat 1D list
    :param _input_file: [file] input file
    :param _output_file: [file] output file
    """
    lines = _input_file.readlines()
    for i in range(len(lines)):
        if lines[i].rstrip() != "":
            word = lines[i].strip()
            # if word == '\n':  # If line is a blank line
            #     output_file.write(word) # Continue inserting blank line
            # else:
            output_file.write("{} {}\n".format(word, _predicted_Y_seqs[i]))

        else:  # Blank line, continue to write blank line
            output_file.write('\n')

    print('Finished writing to File.')


if __name__ == "__main__":
    k = 0.5
    start_time = time.time()
    file_path = sys.argv[1]

    # Get calculated Emission Parameters
    with open("{}/train".format(file_path), "r", encoding="utf8") as training_file:
        emission_params, train_tag_count_dict, train_word_count_dict = EM.get_fixed_emission_params(training_file, 0.5)

    # Get calculated Transition Parameters
    with open("{}/train".format(file_path), "r", encoding="utf8") as training_file:
        transition_params = get_transition_params(training_file)
        print(transition_params)

    tags_list = list(transition_params.keys())
    tags_list.remove("START")
    tags_list.remove("STOP")

    cur_max_count = 0
    common = ""
    for tag, count in train_tag_count_dict.items():
        if count > cur_max_count:
            common = tag
            cur_max_count = count
    unknown_idx = tags_list.index(common)

    # Get Predicted Y Sequences from Viterbi Algorithm
    with open("{}/dev.in".format(file_path), "r", encoding="utf8") as input_file:
        predicted_Y_seqs = get_viterbi_predictions(input_file, emission_params, transition_params, tags_list)
        print(predicted_Y_seqs)


    # Write Predicted Y Sequences to output file
    with open("{}/dev.p3.out".format(file_path), "w+", encoding="utf8") as output_file:
        with open("{}/dev.in".format(file_path), "r", encoding="utf8") as input_file:
            generate_predictions(predicted_Y_seqs, input_file, output_file)

    stop_time = time.time()
    print("Time taken:", stop_time - start_time)
