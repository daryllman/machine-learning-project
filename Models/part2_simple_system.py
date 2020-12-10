import sys
from collections import defaultdict
import time

# ===================================================================================================================
# Write a function that estimates the emission parameters from training set using MLE (Maximum Likelihood Estimation)
# e(x|y) = [ Count(y->x) / Count(y) ]
# ___________________________________________________________________________________________________________________
def get_tag_word_counts(file):
    """
    :param file: [file] file with word and tag in each line
    :return: [dict] Store tag & word counts in the form of dictionary[y][x] where x=word and y=tag,
             [dict] Store tag counts,
             [dict] store word counts
    """
    # Dictionaries to store Tag and Word counts
    tag_count_dict = defaultdict(int)
    word_count_dict = defaultdict(int)
    # Dictionary to store count of specific words with specific tags in the form of: dictionary[tag][word]
    tag_word_count_dict = defaultdict(lambda: defaultdict(int))

    lines = [line for line in file.readlines() if line.strip()]  # List of non-empty lines
    for line in lines:
        word, tag = line.split()
        tag_count_dict[tag] += 1  # Increase count for every specific tag
        word_count_dict[word] += 1  # Increase count for every specific word
        tag_word_count_dict[tag][word] += 1  # Increase count for every specific word we encounter with a specific tag
    return tag_word_count_dict, tag_count_dict, word_count_dict


def get_emission_params(file):
    """
    :param file: [file] file with word and tag in each line
    :return: [dict] Emission Parameters in the form of dictionary[y][x] or "e(x/y)" where x=word and y=tag,
             [dict] Store Tag counts,
             [dict] Store word counts
    """
    tag_word_count_dict, tag_count_dict, word_count_dict = get_tag_word_counts(file)

    # Dictionary to store Emission Parameters in the form of: dictionary[y][x] or "e(x/y)" where x=word and y=tag
    emission_parameters = {}

    for tag in tag_count_dict.keys():
        emission_parameters[tag] = {}
        for word in tag_word_count_dict[tag].keys():
            emission_parameters[tag][word] = float(tag_word_count_dict[tag][word] / tag_count_dict[tag])
    return emission_parameters, tag_count_dict, word_count_dict


# ===================================================================================================================
# Modify function to handle words that do not appear in test set with special word token #UNK#
# e(x|y) = [ Count(y->x) / ( Count(y) + k ) ] if the word token x appears in the training set
# e(x|y) = [ k / ( Count(y) + k ) ] if the word token x is the special token
# ___________________________________________________________________________________________________________________
def get_word_counts(file):  # Takes in test file
    """
    :param file: [file] file with word in each line
    :return: [dict] store word counts
    """
    # Dictionaries to store Word counts
    word_count_dict = defaultdict(int)

    words = [word.rstrip() for word in file.readlines() if word.strip()]  # List of non-empty words
    for word in words:
        word_count_dict[word] += 1  # Increase count for every specific word
    return word_count_dict


def get_fixed_emission_params(_train_file, _k):
    """
    :param _train_file: [file] Training File with word and tag in each line
    :param _k: [float] Fixed decimal constant
    :return: [dict] Emission Parameters in the form of dictionary[y][x] or "e(x/y)" where x=word and y=tag,
             [dict] Store Tag counts (training),
             [dict] Store Word counts (training),
    """
    # Get Dictionaries of word and tag counts from test and train files
    emission_params, train_tag_count_dict, train_word_count_dict = get_emission_params(_train_file)
    #test_word_count_dict = get_word_counts(_test_file)
    # Dictionary to store Emission Parameters in the form of: dictionary[y][x] or "e(x/y)" where x=word and y=tag

    # for test_word in test_word_count_dict.keys():
    #     if test_word not in train_word_count_dict:  # If word appear in test set, but not in training set
    #         for train_tag in train_tag_count_dict.keys():
    #             emission_parameters[train_tag]["#UNK#"] = float(_k / (train_tag_count_dict[train_tag] + _k))  # Note: replace word with #UNK#
    #     else:  # If word appears in both test set and training set
    #         for train_tag in train_tag_count_dict.keys():
    #             for train_word in train_word_count_dict.keys():
    #                 emission_parameters[train_tag][train_word] = float(train_tag_word_count_dict[train_tag][train_word] / train_tag_count_dict[train_tag])
    for tag in emission_params.keys():
        emission_params[tag]["#UNK#"] = float(_k / (train_tag_count_dict[tag] + _k))
    return emission_params, train_tag_count_dict, train_word_count_dict


# ===================================================================================================================
# Implement a simple system that produces the tag
# y* = arg-max-y e(x/y)
# ___________________________________________________________________________________________________________________
def get_emission(e_params, _train_word_count_dict, tag, word):
    """
    :param e_params: [dict] Fixed Emission Parameters in the form of dictionary[y][x] or "e(x/y)" where x=word and y=tag,
    :param _train_word_count_dict: [dict] Store Word Counts of training set
    :param tag: [str] Tag
    :param word: [str] Word
    :return: [float] emission value
    """
    if word in _train_word_count_dict:
        # If word exist in training set and emission params, return the saved emission value
        if word in e_params[tag]:
            return e_params[tag][word]
        else:  # If word exist in training set but not saved in emission params, return 0
            return 0
    else:  # If word does not exist in training set at all
        return e_params[tag]["#UNK#"]


# Final function to predict tags and write into a file
def prediction(e_params, _train_tag_count_dict, _train_word_count_dict, _input_file, output_file):
    for line in input_file:
        if line.rstrip() != "":
            word = line.strip()
            emissions_list = []
            for tag in _train_tag_count_dict.keys():
                emissions_list.append((get_emission(e_params, _train_word_count_dict, tag, word), tag))  # Tuple: (Score, Tag)
            print(emissions_list)
            best_emission = max(emissions_list)[1]  # Will check which tuple has highest score, and return its tag
            output_file.write("{} {}\n".format(word, best_emission))
            assert (max(emissions_list)[0] > 0)  # Should not throw error here...
        else:  # Blank line, continue to write blank line
            output_file.write('\n')
    print("Finished Prediction Task")


if __name__ == "__main__":
    k = 0.5
    start_time = time.time()
    file_path = sys.argv[1]
    """
    # Training File to get emission parameters
    with open("{}/train".format(file_path), "r") as training_file:
        # Get Emission Parameters from training
        train_emission_params, train_tag_count_dict, train_word_count_dict = get_emission_params(training_file)  # Redundant.
    """

    # Test File to get emission parameters (fixed)
    with open("{}/train".format(file_path), "r") as training_file:
        fixed_emission_params, train_tag_count_dict, train_word_count_dict = get_fixed_emission_params(training_file, k)

    # Predict and write prediction results
    with open("{}/dev.p2.out".format(file_path), "w+") as output_file:
        with open("{}/dev.in".format(file_path), "r") as input_file:
            prediction(fixed_emission_params, train_tag_count_dict, train_word_count_dict, input_file, output_file)

    stop_time = time.time()
    print("Time taken:", stop_time - start_time)
