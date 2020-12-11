import sys
from collections import defaultdict, Counter
import time
from itertools import groupby
from math import log
from sys import maxsize  # Arbitrary large value

# Local File Import
import part2_simple_system as EM  # Will be using emission parameters function from part 3
import part3_viterbi as TR  # Will be using transmission parameters function from part 4

# ===================================================================================================================
# Implement an algorithm to find the 3rd best output sequences.
# ___________________________________________________________________________________________________________________
def viterbi_kth_best(_x_seq, _emission_params, _transition_params, _tags_list, _k):
    """
    :param _x_seq: [list] List of X sequence (or a sentence)
    :param _emission_params:  [dict] emission parameters
    :param _transition_params: [dict] transition parameters
    :param _tags_list: [list] 2D List of tags in transition parameters (without START and STOP tags)
    :param _k: [int] Kth best integer
    :return: [list] Predicted Y sequences in a flat 1D list
             [list] Pi List
             [list] Args list
    """
    # Initialise some common variables
    n = len(_x_seq)  # Number of words in _x_seq
    pi_list = [[[0] for tag in range(len(_tags_list))] for j in range(n)]  # List of current max scores e.g. [ [[0], [0], [0]], [[0] [0],[0]] ]
    args_list = [[[""]for arg in range(len(_tags_list))] for j in range(n)]  # Arg list e.g. [ [[""], [""], [""]], [[""] [""],[""]] ]

    # ___________________________________________________
    # Forward Algorithm - Left to Right (From START)
    # ___________________________________________________

    # 1. Initialisation - START to first word
    for i in range(len(_tags_list)):
        word = _x_seq[0].strip()

        # Dynamically handle the different cases,e.g. if no "START" in dict
        try:
            tr = _transition_params[_tags_list[i]]["START"]
            #em = EM.get_emission(_emission_params, _x_seq, _tags_list[i], word)  # wrong
            em = EM.get_emission(_emission_params, train_word_count_dict, _tags_list[i], word)  # def get_emission(e_params, _train_word_count_dict, tag, word):

            if em == 0:
                em_score = -maxsize  # Set large constant negative value
            else:
                em_score = log(em)  # Use logarithmic function
            tr_score = log(tr)
            pi_list[0][i] = [tr_score + em_score]

        except KeyError:
            if em == 0:
                em_score = -maxsize
            else:
                em_score = log(em)
            pi_list[0][i] = [-maxsize + em_score]
        except TypeError:
            pi_list[0][i] = [-maxsize]


    # 2. Main Loop
    for i in range(1, n):
        word = _x_seq[i].strip()
        for j in range(len(_tags_list)):
            path_dict = {}

            for k in range(len(_tags_list)):
                for t in range(len(pi_list[i-1][0])):

                    # Dynamically handle the different cases,e.g. if no particular tag in dict
                    try:
                        prev_pi = pi_list[i-1][k][t]

                        tr = _transition_params[_tags_list[j]][_tags_list[k]]
                        # em = EM.get_emission(_emission_params, _x_seq, _tags_list[j], word) # wrong
                        em = EM.get_emission(_emission_params, train_word_count_dict, _tags_list[j], word)  # def get_emission(e_params, _train_word_count_dict, tag, word):

                        if em == 0:
                            em_score = -maxsize  # Set large constant negative value
                        else:
                            em_score = log(em)  # Use logarithmic function

                        curr_val = prev_pi + log(tr) + em_score

                    except KeyError:
                        if em == 0:
                            em_score = -maxsize  # Set large constant negative value
                        else:
                            em_score = log(em)  # Use logarithmic function
                        curr_val = prev_pi - maxsize + em_score

                    path = args_list[i-1][k][t]
                    path_dict[(k+1, path)] = curr_val

            kt = Counter(path_dict)
            top_k = kt.most_common(_k)
            top_k_prob_list = []
            top_k_path_list = []

            for z in range(_k):
                top_k_prob_list.append(top_k[z][1])
                prev_path = top_k[z][0][1]
                new_path_str = prev_path + "," + str(top_k[z][0][0])
                top_k_path_list.append(new_path_str)

            pi_list[i][j] = top_k_prob_list
            args_list[i][j] = top_k_path_list

    # ___________________________________________________
    # Backward Algorithm - Right to Left (From STOP)
    # ___________________________________________________
    y_seq = [0]*n  # To store predicted best Y sequences
    idx_list = [0]*n  # To store index positions

    # 1. Start Backtracking from STOP to last tag
    last_tag_layer = pi_list[n-1]
    final_check = {}

    # Get kth largest - last layer
    for i in range(len(last_tag_layer)):
        for t in range(_k):
            prev_pi = last_tag_layer[i][t]
            # Dynamically handle the different cases
            try:
                tr = _transition_params["STOP"][_tags_list[i]]
                curr_val = prev_pi + log(tr)
            except KeyError:
                curr_val = prev_pi - maxsize
            final_check[(i, t)] = curr_val
    last = Counter(final_check)
    top_last = last.most_common(_k)

    # Get k best paths and store it
    top_k_paths = []
    for z in range(_k):
        i = top_last[z][0][0]
        t = top_last[z][0][1]
        prev_path = args_list[n-1][i][t]
        total_path_str = "{},{}".format(prev_path, str(i+1))
        total_path_tags = []
        total_path_list = total_path_str.split(",")
        total_path_list = total_path_list[1:]

        for j in range(len(total_path_list)):
            total_path_tags.append(_tags_list[int(total_path_list[j])-1])

        top_k_paths.append(total_path_tags)

    return top_k_paths, pi_list, args_list


def get_viterbi_predictions_kth_best(_input_file, _emission_params, _transition_params, _tags_list, _k):
    """
    :param _input_file: [file] input txt file with word and tag
    :param _emission_params: [dict] emission parameters
    :param _transition_params: [dict] transition parameters
    :param _tags_list: [list] 2D List of tags in transition parameters (without START and STOP tags)
    :param _k: [int] Kth best integer
    :return: [list] Predicted Y sequences in a flat 1D list
    """
    lines = _input_file.readlines()
    # x_seqs_list = formatted_x_seqs_list(lines)  # List of X Sequences (or sentences)
    x_seqs_list = [list(group) for k, group in groupby(lines, lambda x: x == "\n") if not k]
    y_seqs_list = []  # Store the predicted y sequences in a list (including empty lines)

    for i in range(len(x_seqs_list)):
        y_seq, pi_list, args_list = viterbi_kth_best(x_seqs_list[i], _emission_params, _transition_params, _tags_list, _k)
        y_seqs_list += y_seq[_k-1]
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
    start_time = time.time()
    file_path = sys.argv[1]

    # Get calculated Emission Parameters
    with open("{}/train".format(file_path), "r", encoding="utf8") as training_file:
        emission_params, train_tag_count_dict, train_word_count_dict = EM.get_fixed_emission_params(training_file, 0.5)

    # Get calculated Transition Parameters
    with open("{}/train".format(file_path), "r", encoding="utf8") as training_file:
        transition_params = TR.get_transition_params(training_file)
        print(transition_params)

    tags_list = list(transition_params.keys())
    tags_list.remove("START")
    tags_list.remove("STOP")

    # Get Predicted Y Sequences from Viterbi Algorithm
    with open("{}/dev.in".format(file_path), "r", encoding="utf8") as input_file:
        predicted_Y_seqs = get_viterbi_predictions_kth_best(input_file, emission_params, transition_params, tags_list, 3)  # 3rd best
        print(predicted_Y_seqs)

    # Write Predicted Y Sequences to output file
    with open("{}/dev.p4.out".format(file_path), "w+", encoding="utf8") as output_file:
        with open("{}/dev.in".format(file_path), "r", encoding="utf8") as input_file:
            generate_predictions(predicted_Y_seqs, input_file, output_file)

    stop_time = time.time()
    print("Time taken:", stop_time - start_time)
