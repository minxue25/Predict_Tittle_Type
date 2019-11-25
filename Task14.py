# -------------------------------------------------------
# Project 2
# Written by Minxue Sun 40084491, Tian Wang 40079289
# For COMP 6721 Section FJ – Fall 2019
# --------------------------------------------------------
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
import math
from decimal import Decimal
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# read file and get data
df = pd.read_csv('hn2018_2019.csv', usecols=['Title', 'Post Type', 'Created At']) #, encoding='latin-1')
# df = pd.read_csv('test2000.csv', usecols=['Title', 'Post Type', 'Created At']) #, encoding='latin-1')
df18 = df[(True ^ df['Created At'].str.contains('2019'))]
df19 = df[(True ^ df['Created At'].str.contains('2018'))]
list_dt18 = np.array(df18)
list_dt19 = np.array(df19)
# df.to_csv("hn2018_2019(new).csv")

# count post_type and the number of post_type
df18['Post Type'].value_counts()
types = df18['Post Type'].value_counts()
type_dict = {}  # {post_type name:[index, total_num], ...}

post_type_num = len(types)
for i in range(post_type_num):
    type_dict[types.index[i]] = [i, types[i]]

punctuation_table = ["!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", ":", ";",
                     "<", "=", ">", "?", "@", "[", "\\", "]", "^", "`", "{", "|", "}", "~",
                     "﻿", "/", ".", "-", "_", "，", "≈", "≥", "≤", "≠", "⋆", "⋅", "⋙", "√",
                     "∞", "‘", "’", "“", "”", "„", "•", "…", "‹", "›", "→", "↔", "⇆", "−",
                     "∘", "∙", "∩", "─"]

number_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

letter_table = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

remove_set = set()


#  split '(or others) from 'word and become ["'","word"]
def split_word(word_list):
    new_word_list = []
    append_new = new_word_list.append
    for item in word_list:
        item = item.strip()
        if item.startswith("'") and (len(item) > 1):
            append_new("'")
            append_new(item[1:])
        elif item.startswith("~") and (len(item) > 1):
            append_new("~")
            append_new(item[1:])
        elif item.startswith("|") and (len(item) > 1):
            append_new("|")
            append_new(item[1:])
        elif item.startswith("，") and (len(item) > 1):
            append_new("，")
            append_new(item[1:])
        elif item.startswith("≈") and (len(item) > 1):
            append_new("≈")
            append_new(item[1:])
        elif item.startswith("﻿") and (len(item) > 1):
            # append_new("﻿")
            append_new(item[1:])
        elif item.startswith("﻿﻿") and (len(item) > 2):
            # append_new("﻿﻿")
            append_new(item[2:])
        elif item is not "":
            append_new(item)
    return new_word_list


def split_cd(word):
    split_word = []
    start = 0
    for index0 in range(len(word)):
        start_character = word[index0]
        if start_character in (number_table + punctuation_table):
            start = index0 + 1
        else:
            break
    split_word.append(word[0:start])
    split_word.append(word[start:])
    return split_word


def check_all(word, table):
    is_in_table = True
    for index0 in range(len(word)):
        character = word[index0]
        if character in table:
            continue
        else:
            is_in_table = False
            break
    return is_in_table


def check_exist(word, table):
    is_in_table = False
    for index0 in range(len(word)):
        character = word[index0]
        if character in table:
            is_in_table = True
            break
        else:
            continue
    return is_in_table


#  create a new list that has the final token we tokenize in the title
def recreate_list(word_list, save_in_remove_file):
    new_list = []
    append_recreat = new_list.append
    prior_is_noun = False
    for index in range(len(word_list)):  # one tuple: (word, pos)
        if prior_is_noun is False:
            item = word_list[index]
            if item[1] in "CD":
                ordinal_numeral = check_all(item[0], number_table + punctuation_table)
                if ordinal_numeral:
                    if save_in_remove_file:
                        remove_set.add(item[0])
                else:
                    append_recreat(item[0])
            elif item[1] is "JJ":
                ordinal_numeral = check_all(item[0], number_table + punctuation_table)
                if ordinal_numeral:
                    if save_in_remove_file:
                        remove_set.add(item[0])
                else:
                    append_recreat(item[0])
            elif item[1] in ["JJR", "JJS"]:  # adj
                token = lemmatizer.lemmatize(item[0], 'a')
                append_recreat(token)
            elif item[1] in ["RBR", "RBS"]:  # adv
                token = lemmatizer.lemmatize(item[0], 'r')
                append_recreat(token)
            elif item[1] in ["VBD", "VBG", "VBN", "VBP", "VBZ"]:  # verb
                token = lemmatizer.lemmatize(item[0], 'v')
                append_recreat(token)
            elif item[1] in ["NNS"]:  # noun
                token = lemmatizer.lemmatize(item[0], 'n')
                append_recreat(token)
            elif item[1] in ["NNP", "NNPS"]:  # noun
                is_word = item[0].isalpha()
                token = lemmatizer.lemmatize(item[0], 'n')
                if (index + 1) < len(word_list) and is_word:
                    item1 = word_list[index+1]
                    is_word1 = item1[0].isalpha()
                    if (item1[1] in ["NNP", "NNPS"]) and is_word1:
                        token1 = lemmatizer.lemmatize(item1[0], 'n')
                        append_recreat(token + " " + token1)
                        prior_is_noun = True
                    else:
                        is_num = check_all(item[0], number_table + punctuation_table)
                        if is_num:
                            if save_in_remove_file:
                                remove_set.add(item[0])
                        elif item1[1] in ["NN"]:  # noun
                            append_recreat(item1[0])
                        elif item1[1] in ["NNS"]:  # noun
                            token = lemmatizer.lemmatize(item1[0], 'n')
                            append_recreat(token)
                else:  # last one
                    is_num = check_all(item[0], number_table + punctuation_table)
                    if is_num:
                        if save_in_remove_file:
                            remove_set.add(item[0])
                    else:
                        append_recreat(item[0])
            else:
                ordinal_numeral = check_all(item[0], number_table + punctuation_table)
                if ordinal_numeral:
                    if save_in_remove_file:
                        remove_set.add(item[0])
                else:
                    append_recreat(item[0])
        elif prior_is_noun is True:
            prior_is_noun = False
    return new_list


def filter_title(title, removefile):
    words = word_tokenize(title)
    words1 = split_word(words)
    words_list = nltk.pos_tag(words1)
    words_list1 = recreate_list(words_list, removefile)
    return words_list1


# create dictionary
def create_dictionary(removefile):
    vdict = {}  # {key(word), [num of type1, num of type2, num of type3, num of type4,... , total num]}
    for obj in list_dt18:
        title = obj[0]   #.lower()
        post_type = obj[1]
        post_type_index = type_dict[post_type][0]  # {key, [index, total_num], ...}
        words_list = filter_title(title, removefile)
        for token in words_list:
            token = token.lower()
            if token in punctuation_table:
                remove_set.add(token)
            else:
                if vdict.get(token) is None:
                    vdict[token] = [0] * (post_type_num + 1)
                list1 = vdict.get(token)
                list1[post_type_index] = list1[post_type_index] + 1
                list1[post_type_num] = list1[post_type_num] + 1
                vdict[token] = list1
    return vdict


def create_vocabulary(vdict, savevocabulary, vocabulary_filename):
    vocabulary = sorted(vdict.keys())
    if savevocabulary:
        s = ""
        i = 1
        output_file = open(vocabulary_filename, 'a')
        for item in vocabulary:
            output_file.write(str(i) + "  " + item + "\n")  # + "  " + str(vdict.get(item)) + "\n")
            i = i + 1
        output_file.close()
    return vocabulary


#  sum total word of each type
def sum_type_value(vdict):
    vocabulary_value_list = []
    append_voca = vocabulary_value_list.append
    for a_list in vdict:
        append_voca(vdict.get(a_list))
    types_value_list = np.sum(vocabulary_value_list, axis=0)
    return types_value_list


def cal_prob(word_num_in_type, smoothed, type_total_num, vocabulary_len):
    probability = float(word_num_in_type + smoothed) / (type_total_num + smoothed * vocabulary_len)
    if probability != 0 or probability != 0.0:
        log_prob = math.log(probability, 10)
        log_prob = Decimal(log_prob).quantize(Decimal("0.0000000000"))
        return log_prob
    else:
        return math.nan


#  get word probability (Task 1)
def get_probability(smooth, vdict, vocabulary, types_value_list, savefile, filename):
    vdict_with_pro = {}  # {key(word):[prob of type1(log10), prob of type2, prob of type3, prob of type4,...], ...}
    vocabulary_len = len(vocabulary)
    if savefile:
        row_num = 1
        output_file = open(filename, 'a')
        output_file.write("counter, word")
        for ii in range(post_type_num):
            output_file.write(", " + types.index[ii] + ", probability")
        output_file.write("\r\n")
    smoothed = []
    vocabulary_len1 = []
    for index in range(post_type_num):
        smoothed.append(smooth)
        vocabulary_len1.append(vocabulary_len)
    for token in vocabulary:
        if savefile:
            output_file.write(str(row_num) + "  " + str(token))
        list_value = map(cal_prob, vdict.get(token), smoothed, types_value_list, vocabulary_len1)
        vdict_with_pro[token] = list(list_value)
        if savefile:
            for index in range(post_type_num):
                output_file.write("  " + str(vdict.get(token)[index]) + "  " + str(vdict_with_pro[token][index]))
            output_file.write("\r\n")
            row_num = row_num + 1
    if savefile:
        output_file.close()
    return vdict_with_pro


def training_data(dictionary, smooth, savefile, filename, removefile, remove_filename, savevocabulary, vocabulary_filename):
    vocabulary = create_vocabulary(dictionary, savevocabulary, vocabulary_filename)
    types_value_list = sum_type_value(dictionary)
    vdict_with_pro = get_probability(smooth, dictionary, vocabulary, types_value_list, savefile, filename)
    if removefile:
        output_file = open(remove_filename, 'a')
        for item in remove_set:
            output_file.write(item + "\n")
        output_file.close()
    return vdict_with_pro


# score title
def cal_score(tittle_word_list, total_doc_len, vdict_with_prob):
    words_prob = [0] * post_type_num
    for index0 in range(post_type_num):
        words_prob[index0] = math.log(type_dict.get(types.index[index0])[1] / total_doc_len, 10)
        words_prob[index0] = float(Decimal(words_prob[index0]).quantize(Decimal("0.0000000000")))
    for token in tittle_word_list:
        token = token.lower()
        if vdict_with_prob.get(token) is not None:
            words_prob = list(map(lambda x, y: x + float(y), words_prob, vdict_with_prob.get(token)))
        else:
            continue
    return words_prob


def predict_type(prob):
    index = prob.index(max(prob))
    return types.index[index]


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def filter_testing_data(list_dt2019):
    testing_data19 = []
    append0 = testing_data19.append
    for obj in list_dt2019:
        title = obj[0]
        # post_type = obj[1]
        append0(filter_title(title, False))
    return testing_data19


#  testing data and get metrics of results
def testing_data(vdict_with_pro1, savefile, filename, print_result):
    doc_len = len(list_dt18)
    confusion_matrix = np.zeros([post_type_num, post_type_num], dtype=int)
    result_row_num = 0
    output_file = open(filename, 'a')
    if savefile:
        output_file.write("counter,  title,  predict classification")
        for i in range(post_type_num):
            output_file.write(",  " + types.index[i])
        output_file.write(",  correct classification,  result\r\n")
    for obj in list_dt19:
        title = obj[0]
        post_type = obj[1]
        words_list = testing_data2019[result_row_num]#filter_title(title, False)
        probs = cal_score(words_list, doc_len, vdict_with_pro1)
        predict_title_type = predict_type(probs)
        predict_index = type_dict[predict_title_type][0]
        real_index = type_dict[post_type][0]
        confusion_matrix[real_index][predict_index] = confusion_matrix[real_index][predict_index] + 1
        result_row_num = result_row_num + 1
        if savefile:
            result = 'right'
            if predict_title_type != post_type:
                result = 'wrong'
            output_file.write(str(result_row_num) + "  " + str(title) + "  " + str(predict_title_type))
            for index in range(post_type_num):
                output_file.write("  " + str(probs[index]))
            output_file.write("  " + str(post_type) + "  " + str(result) + "\r\n")
    if savefile:
        output_file.close()
    predict_list = np.sum(confusion_matrix, axis=0)
    real_list = np.sum(confusion_matrix, axis=1)
    total_number = np.sum(real_list)

    if print_result:
        matrix = "\n++++++++++++++++ Confusion Matrix/Contingency Table: ++++++++++++++++\nreal|predict"
        for i in range(post_type_num):
            if i is 0:
                matrix = matrix + "\t" + types.index[i]
            else:
                matrix = matrix + "\t\t" + types.index[i]
        matrix = matrix + "\t\ttotal\r\n"
        for real in range(post_type_num):
            matrix = matrix + types.index[real]
            for pred in range(post_type_num):
                matrix = matrix + "\t\t" + str(confusion_matrix[real][pred])
            matrix = matrix + "\t\t" + str(real_list[real]) + "\r\n"
        matrix = matrix + "total"
        for index in range(len(predict_list)):
            matrix = matrix + "\t\t" + str(predict_list[index])
        matrix = matrix + "\t\t" + str(total_number) + "\n"
        print(matrix)

    right = 0
    for ii0 in range(post_type_num):
        right = right + confusion_matrix[ii0][ii0]
    precision = [math.nan] * post_type_num
    recall = [math.nan] * post_type_num
    f1_measure = [math.nan] * post_type_num
    accuracy = right / total_number
    for ii1 in range(post_type_num):
        if predict_list[ii1] is not 0:
            precision[ii1] = confusion_matrix[ii1][ii1] / predict_list[ii1]
        if real_list[ii1] is not 0:
            recall[ii1] = confusion_matrix[ii1][ii1] / real_list[ii1]
        if (precision[ii1] is not math.nan) and (recall[ii1] is not math.nan):
            f1_measure[ii1] = (2 * precision[ii1] * recall[ii1]) / (precision[ii1] + recall[ii1])
    if print_result:
        print("Accuracy: \t", accuracy)
        print("Precision: \t", precision)
        print("Recall: \t", recall)
        print("F1_Measure: ", f1_measure)
    return[accuracy, precision, recall, f1_measure]


def optimize_dic_stop_word(dictionary):
    f = open('Stopwords.txt', 'r')
    result = []
    append_stop = result.append
    for line in open('Stopwords.txt'):
        line = f.readline()
        append_stop(line.rstrip('\n'))
    f.close()
    for item in result:
        if dictionary.get(item) is not None:
            del (dictionary[item])
        else:
            continue


def training_testing_data(dic, smooth, savemodel, model_filename, saveremove, remove_filename, saveresult, result_filename, savevocabulary, vocabulary_filename, print_result):
    start_time = time.time()
    vdict_with_pro = training_data(dic, smooth, savemodel, model_filename, saveremove, remove_filename, savevocabulary, vocabulary_filename)
    if print_result:
        time1 = time.time()
        print("Training data cost:", time1 - start_time, "s")
    # Test
    start_time = time.time()
    result = testing_data(vdict_with_pro, saveresult, result_filename, print_result)
    if print_result:
        time1 = time.time()
        print("\nTesting data cost: ", time1-start_time, "s")
    return result


def optimize_dic_word_length(dictionary, lessthan, greaterthan):
    remove_list = []
    append_length = remove_list.append
    for item in dictionary:
        if (len(item) <= lessthan) or (len(item) >= greaterthan):
            append_length(item)
    for item in remove_list:
        del(dictionary[item])


def optimize_dic_infrequent_word(dictionary, less, infrequent, type_num):
    remove_list = []
    append_fre = remove_list.append
    if less is True:
        for item in dictionary:
            if dictionary.get(item)[type_num] <= infrequent:
                append_fre(item)
    else:
        for item in dictionary:
            if dictionary.get(item)[type_num] >= infrequent:
                append_fre(item)
    for item in remove_list:
        del (dictionary[item])


def change_infrequent(dic3_3, less, frequent, savevoca, vocafilename, print_result):
    start_time = time.time()
    if print_result:
        print("\n***[Start frequent", frequent,"]***")
    optimize_dic_infrequent_word(dic3_3, less, frequent, post_type_num)
    if print_result:
        time1 = time.time()
        print("Create vocabulary cost: ", time1 - start_time, "s")
    result = training_testing_data(dic3_3, 0.5, False, 'infrequent-model3_3.txt', False, 'remove_words3_3.txt', False, 'infrequent-result.txt', savevoca, vocafilename, print_result)
    return result


def cal_frequent(percentage, v_list):
    limit_frequent = 0
    if (v_list is not None) and (len(v_list) > 0):
        index = math.floor(percentage * len(v_list))
        limit_frequent = v_list[index]
    return limit_frequent


# plot a graph of results
def plot_graph(title, x_lable, x_alias, result_list):
    fig_row = math.ceil(post_type_num/2)
    plt.figure(str(fig_row)+"2", figsize=(16, 10))
    plt.ylim(0, 100)
    y = np.arange(0, 100, 10)
    x = np.arange(1, len(result_list) + 1, 1)
    for i in range(post_type_num):
        sub_position = str(fig_row)+"2"+str(i+1)
        plt.subplot(sub_position)
        list_accuracy = []
        list_precision = []
        list_recall = []
        list_f1_measure = []
        for i0 in range(len(result_list)):
            item = result_list[i0]
            list_accuracy.append(item[0])
            list_precision.append(item[1][i])
            list_recall.append(item[2][i])
            list_f1_measure.append(item[3][i])
        list_accuracy = [i * 100 for i in list_accuracy]
        list_precision = [i * 100 for i in list_precision]
        list_recall = [i * 100 for i in list_recall]
        list_f1_measure = [i * 100 for i in list_f1_measure]
        plt.plot(x, list_accuracy, 'y--', label='accuracy')
        plt.plot(x, list_precision, 'r--', label='precision')
        plt.plot(x, list_recall, 'g--', label='recall')
        plt.plot(x, list_f1_measure, 'b--', label='f1-measure')
        plt.plot(x, list_accuracy, 'y.-', x, list_precision, 'ro-', x, list_recall, 'g+-', x, list_f1_measure, 'b^-')
        plt.title('The results in '+types.index[i]+" (different " + title + ")")
        plt.xlabel(x_lable)
        plt.ylabel('%')
        plt.xticks(x, x_alias)
        plt.legend()
    plt.show()


def change_smooth(smooth, savevoca, vocafilename, print_result):
    if print_result:
        print('\n***[Start smooth', smooth,"]***")
    result = training_testing_data(dictionary, smooth, False, 'smooth-model.txt', False, 'remove_words3_4.txt', False, 'smooth-result.txt', savevoca, vocafilename, print_result)
    return result


# Task1
print("================================== Task 1 =========================================================")
print('Start filter training data...')
start_time = time.time()
dictionary = create_dictionary(True)
time1 = time.time()
print("Create vocabulary cost: ", time1 - start_time, "s")
vdict_with_pro = {}

print('Start filter testing data...')
start_time = time.time()
testing_data2019 = filter_testing_data(list_dt19)
time1 = time.time()
print("filter testing data cost: ", time1 - start_time, "s")


print("Start training data...")
start_time = time.time()
vdict_with_pro = training_data(dictionary, 0.5, True, 'model-2018.txt', True, 'remove_words.txt', True, "vocabulary.txt")
time1 = time.time()
print("Training data cost: ", time1 - start_time, "s")
print("Task1 Completed! Please read 'model-2018.txt', 'vocabulary.txt' and 'remove_words.txt' ^u^ ")
print("===================================================================================================\n")

# Task2
print("================================== Task 2 =========================================================")
task2 = input("Please press keyboard to continue Task2: (1 is print result, else not)")
if task2 is not None:
    print_result = False
    if task2 is "1":
        print_result = True
    print('Start Task2...')
    start_time = time.time()
    result = testing_data(vdict_with_pro, True, 'baseline-result.txt', print_result)
    time1 = time.time()
    print("\nTesting data cost: ", time1-start_time, "s")
    print("Task2 Completed! Please read 'baseline-result.txt' ^u^")
print("===================================================================================================\n")


#  Task3.1
print("================================== Task 3.1 =======================================================")
task3_1 = input("Please press keyboard to continue 3.1: (1 is print result, else not)")
if task3_1 is not None:
    print_result = False
    if task3_1 is "1":
        print_result = True
    print('Start Task3.1...')
    start_time = time.time()
    dic3_1 = dictionary.copy()
    optimize_dic_stop_word(dic3_1)
    # optimize_dic_word_length(dic3_1, 2, 999)
    time1 = time.time()
    print("Create vocabulary cost: ", time1 - start_time, "s")
    result = training_testing_data(dic3_1, 0.5, True, 'stopword-model.txt', False, 'remove_words3_1.txt', True, 'stopword-result.txt', False, "vocabulary3_1.txt", print_result)
    print("Task3.1 Completed! Please read 'stopword-model.txt' ^u^")
print("===================================================================================================\n")


# Task3.2
print("================================== Task 3.2 =======================================================")
task3_2 = input("Please press keyboard to continue 3.2: (1 is print result, else not)")
if task3_2 is not None:
    print_result = False
    if task3_2 is "1":
        print_result = True
    print('Start Task3.2...')
    start_time = time.time()
    dic3_2 = dictionary.copy()
    optimize_dic_word_length(dic3_2, 2, 9)
    time1 = time.time()
    print("Create vocabulary cost: ", time1 - start_time, "s")
    result = training_testing_data(dic3_2, 0.5, True, 'wordlength-model.txt', False, 'remove_words3_2.txt', True, 'wordlength-result.txt', False, "vocabulary3_2.txt", print_result)
    print("Task3.2 Completed! Please read 'wordlength-model.txt' ^u^")
print("===================================================================================================\n")


#Task3.3
# frequency = 1, frequency ≤ 5, frequency ≤ 10, frequency ≤ 15 and frequency ≤ 20
# frequency >= 5%, frequency >= 15%, frequency >= 20%, frequency >= 25%
print("================================== Task 3.3 =======================================================")
task3_3 = input("Please press keyboard to continue Task3.3: (1 is print result, else not)")
if task3_3 is not None:
    print_result = False
    if task3_3 is "1":
        print_result = True
    start_time = time.time()
    print('Start Task3.3...')
    analyse_fre_result_list = []
    append1 = analyse_fre_result_list.append
    dic3_3_1 = dictionary.copy()
    append1(change_infrequent(dic3_3_1, True, 1, False, "vocabulary3_3(1).txt", print_result))
    x1 = len(dic3_3_1)
    append1(change_infrequent(dic3_3_1, True, 5, False, "vocabulary3_3(5).txt", print_result))
    x2 = len(dic3_3_1)
    append1(change_infrequent(dic3_3_1, True, 10, False, "vocabulary3_3(10).txt", print_result))
    x3 = len(dic3_3_1)
    append1(change_infrequent(dic3_3_1, True, 15, False, "vocabulary3_3(15).txt", print_result))
    x4 = len(dic3_3_1)
    append1(change_infrequent(dic3_3_1, True, 20, False, "vocabulary3_3(20).txt", print_result))
    x5 = len(dic3_3_1)
    frequent_list = []
    for item in dictionary:
        frequent_list.append(dictionary.get(item)[post_type_num])
    frequent_list.sort(reverse=True)
    dic3_3_2 = dictionary.copy()
    if frequent_list is not None and (len(frequent_list) > 0):
        frequent1 = cal_frequent(0.05, frequent_list)
        append1(change_infrequent(dic3_3_2, False, frequent1, False, "vocabulary3_3(top5).txt", print_result))
        x6 = len(dic3_3_2)
        frequent2 = cal_frequent(0.10, frequent_list)
        append1(change_infrequent(dic3_3_2, False, frequent2, False, "vocabulary3_3(top10).txt", print_result))
        x7 = len(dic3_3_2)
        frequent3 = cal_frequent(0.15, frequent_list)
        append1(change_infrequent(dic3_3_2, False, frequent3, False, "vocabulary3_3(top15).txt", print_result))
        x8 = len(dic3_3_2)
        frequent4 = cal_frequent(0.2, frequent_list)
        append1(change_infrequent(dic3_3_2, False, frequent4, False, "vocabulary3_3(top20).txt", print_result))
        x9 = len(dic3_3_2)
        frequent5 = cal_frequent(0.25, frequent_list)
        append1(change_infrequent(dic3_3_2, False, frequent5, False, "vocabulary3_3(top25).txt", print_result))
        x10 = len(dic3_3_2)
    # x = ["<=1", "<=5", "<=10", "<=15", "<=20", "top 5%("+str(frequent1)+")", "top 10% ("+str(frequent2)+")", "top 15% ("+str(frequent3)+")", "top 20% ("+str(frequent4)+")", "top 25% ("+str(frequent5)+")"]
    x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    plot_graph("frequency", "words left in vocabulary", x, analyse_fre_result_list)
    end_time = time.time()
    print('\nTask3.3 total cost: ', end_time-start_time, "s")
    print("Task3.3 Completed! Please see the figure! ^u^\n")
print("===================================================================================================\n")


# Task3.4
print("================================== Task 3.4 =======================================================")
task3_4 = input("Please press keyboard to continue Task3.4: (1 is print result, else not)")
if task3_4 is not None:
    print_result = False
    if task3_4 is "1":
        print_result = True
    start_time = time.time()
    print('Start Task3.4...')
    analyse_smooth_result_list = []
    append2 = analyse_smooth_result_list.append
    for i in range(11):
        result_list = change_smooth(i/10, False, "vocabulary3_4(" + str(i/10) + ").txt", print_result)
        append2(result_list)
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plot_graph("smooth", "smooth", x, analyse_smooth_result_list)
    time1 = time.time()
    print("\nTask3.4 total cost: ", time1 - start_time, "s")
    print("Task3.4 Completed! Please see the figure! ^u^\n")
print("===================================================================================================\n")
