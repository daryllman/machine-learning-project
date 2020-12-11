from collections import defaultdict
import sys
import os
import random
import string


class WordTagger:
    # Initialises a WordTagger instance with required dictionaries
    def __init__(self, tag_counts):
        self.tag_counts = tag_counts
        self.classes = tag_counts.keys()  # get keys
        self.weights = defaultdict(lambda: defaultdict(int))

    # features here represent a word, we created a list of features to represent each word
    def predict(self, features):
        if features != "":
            # total score keep track of the score for each tag related to the word
            scores = defaultdict(int)
            for feature in features:
                # if the particular feature is not found in our model's weights class, we ignore
                if feature not in self.weights:
                    continue
                # else if it is found, select the dictionary for the feature from the weights class
                weights = self.weights[feature]
                # we ultimately want to use the tag with the highest score as our prediction, thus we increment it and keep track in a dictionary called scores
                for tag, weight in weights.items():
                    scores[tag] += weight
            # we return the tag with the highest score as our prediction
            return max(self.classes, key=lambda tag: (scores[tag], tag))
        return ""

    def train(self, iter, document):
        for i in range(iter):
            print("Training for iteration...", i)


            # pass in the features and correct tag to predict function
            for features, correct_tag in document:
                predicted_tag = self.predict(features)
                # update weights only if there is any wrong prediction
                if predicted_tag != correct_tag:
                    for feature in features:
                        self.weights[feature][predicted_tag] -= 1
                        self.weights[feature][correct_tag] += 1
            random.shuffle(document)

        #averaging perceptron weights
        for feature in self.weights:
            for tag in self.weights[feature]:
                self.weights[feature][tag] = self.weights[feature][tag] / \
                    (iter*len(document))

        return self.weights


def parse_predict_test_file(fileIn, fileOut, model):
    fout = open(fileOut, 'w+', encoding="utf8")
    # list_new_guesses = [""]
    finput = open(fileIn, 'r', encoding="utf8")
    lines = finput.readlines()

    output = []
    for i in range(0, len(lines)):
        # special case for first line: previous and previous previous word features have to be specificially handled since they are null
        if i == 0:
            line = lines[i].strip().split(" ")
            next_line = lines[i+1].strip().split(" ")
            next2_line = lines[i+2].strip().split(" ")
            features = get_features(
                line[0], "", "", "", "", next_line[0], next2_line[0])
            guess = model.predict(features)
            output.append(guess)
            fout.write(line[0]+" "+guess+"\n")
        # special case for second line: previous word features have to be specifically handled since they are null
        elif i == 1:
            line = lines[i].strip().split(" ")
            next_line = lines[i+1].strip().split(" ")
            next2_line = lines[i+2].strip().split(" ")
            prev_line = lines[i-1].strip().split(" ")
            prev_word = prev_line[0]
            prev_tag = output[0]
            features = get_features(
                line[0], prev_word, prev_tag, "", "", next_line[0], next2_line[0])
            guess = model.predict(features)
            output.append(guess)
            fout.write(line[0]+" "+guess+"\n")
        # case for the rest of the document (middle case)
        else:
            if lines[i].strip() != "":
                # read in current current line
                line = lines[i].strip().split(" ")
                # read in previous 2 lines
                prev_line = lines[i-1].strip().split(" ")
                prev2_line = lines[i-2].strip().split(" ")

                # extract features related to current line and previous 2 lines
                word = line[0]
                prev_word = prev_line[0]
                prev_tag = output[i-1]
                prev2_tag = output[i-2]
                prev2_word = prev2_line[0]

                # reading and extracting for next 2 line features
                # special case if we are at the last line of the document:  next word and next next word will be null
                if i == len(lines)-1:
                    next_word = ""
                    next2_word = ""
                # special case if we are at the second last line of the document: next next word will be null
                elif i == len(lines)-2:
                    next_line = lines[i+1].strip().split(" ")
                    next_word = next_line[0]
                    next2_word = ""
                # all other cases in the  middle of the document
                else:
                    next_line = lines[i+1].strip().split(" ")
                    next2_line = lines[i+2].strip().split(" ")
                    next_word = next_line[0]
                    next2_word = next2_line[0]

                # pass in the features to get_features to return a set of feature that we can pass in to predict function
                features = get_features(
                    word, prev_word, prev_tag, prev2_tag, prev2_word, next_word, next2_word)
                guess = model.predict(features)
                output.append(guess)
                fout.write(line[0]+" "+guess+"\n")

            else:
                output.append("")
                fout.write("\n")

    return


def parse_feature_tag_pairs(folder_path, filename):
    output = []
    output.append(("", ""))

    tag_counts = defaultdict(int)
    finput = open(os.path.join(folder_path, filename), 'r', encoding="utf8")
    lines = finput.readlines()

    # handling of special cases is the same as "parse_predict_test_file"
    for i in range(0, len(lines)):
        if i == 0:
            line = lines[i].strip().split(" ")
            next_line = lines[i+1].strip().split(" ")
            next2_line = lines[i+2].strip().split(" ")
            features = get_features(
                line[0], "", "", "", "", next_line[0], next2_line[0])
            output.append((features, line[1]))
        elif i == 1:
            line = lines[i].strip().split(" ")
            next_line = lines[i+1].strip().split(" ")
            prev_line = lines[i-1].strip().split(" ")
            next2_line = lines[i+2].strip().split(" ")
            prev_word = prev_line[0]
            prev_tag = prev_line[1]
            features = get_features(
                line[0], prev_word, prev_tag, "", "", next_line[0], next2_line[0])
            output.append((features, line[1]))

        else:
            if lines[i].strip() != "":
                line = lines[i].strip().split(" ")
                if i == len(lines)-1:
                    next_word = ""
                    next2_word = ""
                elif i == len(lines)-2:
                    next_line = lines[i+1].strip().split(" ")
                    next_word = next_line[0]
                    next2_word = ""
                else:
                    next_line = lines[i+1].strip().split(" ")
                    next_word = next_line[0]
                    next2_line = lines[i+2].strip().split(" ")
                    next2_word = next2_line[0]

                prev_line = lines[i-1].strip().split(" ")
                prev2_line = lines[i-2].strip().split(" ")
                word = line[0]
                tag = line[1]
                prev_word = prev_line[0]
                prev2_word = prev2_line[0]

                if(prev_word == ''):
                    prev_tag = ""
                else:
                    prev_tag = prev_line[1]

                if(prev2_word == ""):
                    prev2_tag = ""
                else:
                    prev2_tag = prev2_line[1]

                features = get_features(
                    word, prev_word, prev_tag, prev2_tag, prev2_word, next_word, next2_word)

                output.append((features, tag))
                tag_counts[tag] += 1
            else:
                output.append(("", ""))

    return output, tag_counts

# Check if first letter is capital
def isFirstCapital(token):
    if token[0].upper() == token[0]:
        return "true"
    else:
        return "false"

# Check if all letters are alpha
def isAlpha(word):
    if word.isalpha():
        return "true"
    else:
        return "false"

# Check if it is stop words
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
def isStopWord(word):
    if word in stopwords:
        return "true"
    else:
        return "false"

# Check length of word
def lenWord(word):
    if len(word) == 1:
        return "very-small"
    elif len(word) <= 3:
        return "small"
    elif len(word) <= 5:
        return "mid"
    elif len(word) <= 7:
        return "mid"
    else:
        return "large"

# If all letters are capitalised
def isUpper(word):
    if word.isupper():
        return "true"
    else:
        return "false"

# If letters are alphanumeric
def isAlphanumeric(word):
    if word.isalnum():
        return "true"
    else:
        return "false"

# If pure digit or digit with fullstop in middle e.g. 2 rabbits & $ 5.0 cake
def isNumber(word):
    try:
        float(word)
        return "true"
    except:
        return "false"



def get_features(word, prev_word, prev_tag, prev2_tag, prev2_word, next_word, next2_word):
    def add(name, *args):
        features.add('+'.join((name,) + tuple(args)))  # generate the features
    # set can only have unique elements.
    features = set()
    add("isFirstCapital", isFirstCapital(word))
    add("isAlpha", isAlpha(word))
    add("isStopWord", isStopWord(word))
    add("lenWord", lenWord(word))
    add("isUpper", isUpper(word))
    add("isAlphanumeric", isAlphanumeric(word))
    add("isNumber", isNumber(word))


    # convert to lower case for better performance
    word = word.lower()
    prev_word = prev_word.lower()
    prev2_word = prev2_word.lower()
    next_word = next_word.lower()
    next2_word = next2_word.lower()

    add('ith-suffix-last-letters', word[-3:])  # suffix of current word
    add('ith-suffix-last-2-letters', word[-2:])  # suffix of current word
    add("ith-prefix", word[0:2])  # prefix of current word
    add("i-1th-tag", prev_tag)
    add("ith-word", word)
    add("i-1th-word", prev_word)
    add("i-2th-word", prev2_word)
    add("i-2th-tag", prev2_tag)
    add("i+1th-word", next_word)
    add("i+2th-word", next2_word)
    add("i-1th-suffix", prev_word[-3:])  # suffix of previous word
    add("i+1th-suffix", next_word[-3:])  # suffix of next word
    return features


# RUNNING THE CODE
if __name__ == "__main__":

    dataset = sys.argv[1]
    fileIn = dataset+"/dev.in"
    fileOut = dataset+"/dev.p5.out"

    output, tag_counts = parse_feature_tag_pairs(dataset, 'train')
    test = WordTagger(tag_counts)

    n = 20 # Set number of iterations to run perceptron
    model_weights = test.train(n, output)
    parse_predict_test_file(fileIn, fileOut, test)
