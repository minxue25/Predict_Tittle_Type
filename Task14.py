import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
import math
from decimal import Decimal
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()
# read file and get data
df = pd.read_csv('hn2018_2019.csv', usecols=['Title', 'Post Type', 'Created At']) #, encoding='latin-1')
# df = pd.read_csv('test2000.csv', usecols=['Title', 'Post Type', 'Created At']) #, encoding='latin-1')
df18 = df[(True ^ df['Created At'].str.contains('2019'))]
df19 = df[(True ^ df['Created At'].str.contains('2018'))]
list_dt18 = np.array(df18)
list_dt19 = np.array(df19)


# count post_type and the number of post_type
df18['Post Type'].value_counts()
types = df18['Post Type'].value_counts()
type_dict = {}  # {post_type name:[index, total_num], ...}

post_type_num = len(types)
for i in range(post_type_num):
    type_dict[types.index[i]] = [i, types[i]]

print("finish reading")

punctuation_table = ["!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", ":", ";",
              "<", "=", ">", "?", "@", "[", "\\", "]", "^", "`", "{", "|", "}", "~", "﻿",
              "/", ".", "-", "_", "，", "≈", "≥", "≤", "≠", "⋆", "⋅", "⋙", "√", "∞"]

number_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

letter_table = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

remove_set = set()


#  split ' from 'word and become ["'","word"]
def split_word(word_list):
    new_word_list = []
    for item in word_list:
        item = item.strip()
        if item.startswith("'") and (len(item) > 1):
            new_word_list.append("'")
            new_word_list.append(item[1:])
        elif item.startswith("~") and (len(item) > 1):
            new_word_list.append("~")
            new_word_list.append(item[1:])
        elif item.startswith("|") and (len(item) > 1):
            new_word_list.append("|")
            new_word_list.append(item[1:])
        elif item.startswith("，") and (len(item) > 1):
            new_word_list.append("，")
            new_word_list.append(item[1:])
        elif item.startswith("≈") and (len(item) > 1):
            new_word_list.append("≈")
            new_word_list.append(item[1:])
        elif item.startswith("﻿") and (len(item) > 1):
            # new_word_list.append("﻿")
            new_word_list.append(item[1:])
        elif item.startswith("﻿﻿") and (len(item) > 2):
            # new_word_list.append("﻿﻿")
            new_word_list.append(item[2:])
        elif item is not "":
            new_word_list.append(item)
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


def recreate_list(word_list, save_in_remove_file):
    new_list = []
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
                    new_list.append(item[0])
                # number_word = split_cd(item[0])
                # if (number_word[0] is not "") and save_in_remove_file:
                #     remove_set.add(number_word[0])
                # if number_word[1] is not "":
                #     new_list.append(number_word[1])
            elif item[1] is "JJ":
                ordinal_numeral = check_all(item[0], number_table + punctuation_table)
                if ordinal_numeral:
                    if save_in_remove_file:
                        remove_set.add(item[0])
                else:
                    new_list.append(item[0])
            elif item[1] in ["JJR", "JJS"]:  # adj
                token = lemmatizer.lemmatize(item[0], 'a')
                new_list.append(token)
            elif item[1] in ["RBR", "RBS"]:  # adv
                token = lemmatizer.lemmatize(item[0], 'r')
                new_list.append(token)
            elif item[1] in ["VBD", "VBG", "VBN", "VBP", "VBZ"]:  # verb
                token = lemmatizer.lemmatize(item[0], 'v')
                new_list.append(token)
            elif item[1] in ["NNS"]:  # noun
                token = lemmatizer.lemmatize(item[0], 'n')
                new_list.append(token)
            elif item[1] in ["NNP", "NNPS"]:  # noun
                is_word = check_exist(item[0], letter_table)
                token = lemmatizer.lemmatize(item[0], 'n')
                if (index + 1) < len(word_list) and is_word:
                    item1 = word_list[index+1]
                    is_word1 = check_exist(item1[0], letter_table)
                    if (item1[1] in ["NNP", "NNPS"]) and is_word1:
                        token1 = lemmatizer.lemmatize(item1[0], 'n')
                        new_list.append(token + " " + token1)
                        prior_is_noun = True
                    else:
                        is_num = check_all(item[0], number_table + punctuation_table)
                        if is_num:
                            if save_in_remove_file:
                                remove_set.add(item[0])
                        elif item1[1] in ["NN"]:  # noun
                            new_list.append(item1[0])
                        elif item1[1] in ["NNS"]:  # noun
                            token = lemmatizer.lemmatize(item1[0], 'n')
                            new_list.append(token)
                else:  # last one
                    is_num = check_all(item[0], number_table + punctuation_table)
                    if is_num:
                        if save_in_remove_file:
                            remove_set.add(item[0])
                    else:
                        new_list.append(item[0])
            else:
                ordinal_numeral = check_all(item[0], number_table + punctuation_table)
                if ordinal_numeral:
                    if save_in_remove_file:
                        remove_set.add(item[0])
                else:
                    new_list.append(item[0])
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
    i = 0
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


def write_file(string, filename):
    output_file = open(filename, 'w')
    output_file.write(string)
    output_file.close()


def create_vocabulary(vdict, savevocabulary, vocabulary_filename):
    vocabulary = sorted(vdict.keys())
    if savevocabulary:
        s = ""
        i = 1
        for item in vocabulary:
            s += str(i) + "  " + item + "\n"
            i = i + 1
        write_file(s, vocabulary_filename)
    return vocabulary


#  sum total word of each type
def sum_type_value(vdict):
    vocabulary_value_list = []
    for a_list in vdict:
        vocabulary_value_list.append(vdict.get(a_list))
    types_value_list = np.sum(vocabulary_value_list, axis=0)
    return types_value_list


def cal_prob(word_num_in_type, smoothed, type_total_num, vocabulary_len):
    probability = float(word_num_in_type + smoothed) / (type_total_num + smoothed * vocabulary_len)
    if probability is not 0:
        log_prob = math.log(probability, 10)
        log_prob = Decimal(log_prob).quantize(Decimal("0.0000000000"))
        return log_prob
    else:
        return math.nan


# get word probability (Task 1)
def get_probability(smooth, vdict, vocabulary, types_value_list, savefile, filename):
    vdict_with_pro = {}  # {key(word):[prob of type1(log10), prob of type2, prob of type3, prob of type4,...], ...}
    row_num = 1
    vocabulary_len = len(vocabulary)
    s = "counter"
    for ii in range(post_type_num):
        s = s + "\t\t\t" + types.index[ii]
    s = s + "\r\n"
    for token in vocabulary:
        s = s + str(row_num) + "  " + str(token)
        list_value = []
        for index in range(post_type_num):
            word_prob = cal_prob(vdict.get(token)[index], smooth, types_value_list[index], vocabulary_len)
            if savefile:
                s = s + "  " + str(vdict.get(token)[index]) + "  " + str(word_prob)
            list_value.append(word_prob)
        vdict_with_pro[token] = list_value
        # print(token + " " + str(list_value))
        s = s + "\r\n"
        row_num = row_num + 1
    if savefile:
        write_file(s, filename)
    return vdict_with_pro


def training_data(dictionary, smooth, savefile, filename, removefile, remove_filename, savevocabulary, vocabulary_filename):
    print("testing data...")
    vocabulary = create_vocabulary(dictionary, savevocabulary, vocabulary_filename)
    types_value_list = sum_type_value(dictionary)
    vdict_with_pro = get_probability(smooth, dictionary, vocabulary, types_value_list, savefile, filename)
    if removefile:
        remove_word = ""
        for item in remove_set:
            remove_word = remove_word + item + "\n"
        write_file(remove_word, remove_filename)
    return vdict_with_pro


# Task1
start_time = time.time()
print('start task1..')
dictionary = create_dictionary(True)
time1 = time.time()
print("finish saving in dic: ", time1 - start_time, "s")

vdict_with_pro = training_data(dictionary, 0.5, True, 'model-2018.txt', True, 'remove_words.txt', True, "vocabulary.txt")
time1 = time.time()
print("finish pro: ", time1 - start_time, "s")
print("Please read 'model-2018.txt', 'vocabulary.txt' and 'remove_words.txt' ^u^\n")


# Task2
def cal_score(tittle_word_list, total_doc_len, vdict_with_pro):
    words_prob = [0] * post_type_num
    for index0 in range(post_type_num):
        words_prob[index0] = math.log(type_dict.get(types.index[index0])[1] / total_doc_len, 10)
        words_prob[index0] = float(Decimal(words_prob[index0]).quantize(Decimal("0.0000000000")))
    for token in tittle_word_list:
        token = token.lower()
        if vdict_with_pro.get(token) is not None:
            for index1 in range(post_type_num):
                token_prob = vdict_with_pro.get(token)  # {key, [log, log...]..}
                words_prob[index1] = words_prob[index1] + float(token_prob[index1])
        else:
            continue
    return words_prob


def predict_type(prob):
    index = prob.index(max(prob))
    return types.index[index]


def testing_data(vdict_with_pro, savefile, filename):
    doc_len = len(list_dt18)
    confusion_matrix = np.zeros([post_type_num, post_type_num], dtype=int)
    result_row_num = 1
    s = "counter"
    for i in range(post_type_num):
        s = s + "\t" + types.index[i]
    s = s + "\r\n"
    for obj in list_dt19:
        title = obj[0]
        post_type = obj[1]
        words_list = filter_title(title, False)
        probs = cal_score(words_list, doc_len, vdict_with_pro)
        predict_title_type = predict_type(probs)
        predict_index = type_dict[predict_title_type][0]
        real_index = type_dict[post_type][0]
        confusion_matrix[real_index][predict_index] = confusion_matrix[real_index][predict_index] + 1
        if savefile:
            result = 'right'
            if predict_title_type != post_type:
                result = 'wrong'
            s = s + str(result_row_num) + "  " + str(title) + "  " + str(predict_title_type)
            for index in range(len(probs)):
                s = s + "  " + str(probs[index])
            s = s + "  " + str(post_type) + "  " + str(result) + "\r\n"
        result_row_num = result_row_num + 1
    if savefile:
        write_file(s, filename)
    predict_list = np.sum(confusion_matrix, axis=0)
    real_list = np.sum(confusion_matrix, axis=1)
    total_number = np.sum(real_list)

    matrix = "confusion matrix/contingency table:\nreal|predict"
    for i in range(post_type_num):
        matrix = matrix + " " + types.index[i]
    matrix = matrix + "\ttotal\r\n"
    for real in range(post_type_num):
        matrix = matrix + types.index[real]
        for pred in range(post_type_num):
            matrix = matrix + "\t" + str(confusion_matrix[real][pred])
        matrix = matrix + "\t" + str(real_list[real]) + "\r\n"
    matrix = matrix + "total"
    for index in range(len(predict_list)):
        matrix = matrix + "\t" + str(predict_list[index])
    matrix = matrix + "\t" + str(total_number)
    print(matrix)

    right = 0
    for ii0 in range(post_type_num):
        right = right + confusion_matrix[ii0][ii0]
    precision = [0] * post_type_num
    recall = [0] * post_type_num
    f1_measure = [0] * post_type_num
    accuracy = right / total_number
    for ii1 in range(post_type_num):
        precision[ii1] = confusion_matrix[ii1][ii1] / predict_list[ii1]
        recall[ii1] = confusion_matrix[ii1][ii1] / real_list[ii1]
        f1_measure[ii1] = (2 * precision[ii1] * recall[ii1]) / (precision[ii1] + recall[ii1])

    print("Accuracy", accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("F1_Measure", f1_measure)
    return[accuracy, precision, recall, f1_measure]




# Task2
task2 = input("Please press keyboard to continue Task2:")
if task2 is not None:
    start_time = time.time()
    print("start task2...")
    result = testing_data(vdict_with_pro, True, 'baseline-result.txt')
    write_file(str(result), "result_analysis_task2.txt")
    time1 = time.time()
    print("finish task2: ", time1-start_time, "s")
    print("Please read 'baseline-result.txt' (and 'result_analysis_task2.txt') ^u^\n")


# Task3
# Task3.1 Stop-word Filtering
# stopword - model.txt and stopword - result.txt
def optimize_dic_stop_word(dictionary):
    f = open('Stopwords.txt', 'r')
    result = list()
    for line in open('Stopwords.txt'):
        line = f.readline()
        result.append(line.rstrip('\n'))
    f.close()
    for item in result:
        if dictionary.get(item) is not None:
            del (dictionary[item])
        else:
            continue


def training_testing_data(dic, smooth, savemodel, model_filename, saveremove, remove_filename, saveresult, result_filename, savevocabulary, vocabulary_filename ):
    start_time = time.time()
    print("start training data...")
    vdict_with_pro = training_data(dic, smooth, savemodel, model_filename, saveremove, remove_filename, savevocabulary, vocabulary_filename )
    time1 = time.time()
    print("finish pro: ", time1 - start_time, "s")
    # Test
    print("start testing data...")
    result = testing_data(vdict_with_pro, saveresult, result_filename)
    time1 = time.time()
    print("finish task: ", time1-start_time, "s")
    return result


task3_1 = input("Please press keyboard to continue Task3.1:")
if task3_1 is not None:
    print('start3.1..')
    dic3_1 = dictionary.copy()
    optimize_dic_stop_word(dic3_1)
    time1 = time.time()
    print("finish saving in dic: ", time1 - start_time, "s")
    result = training_testing_data(dic3_1, 0.5, True, 'stopword-model.txt', False, 'remove_words3_1.txt', True, 'stopword-result.txt', True, "vocabulary3_1.txt")
    write_file(str(result), "result_analysis_task3_1.txt")
    print("Please read 'stopword-model.txt' ('vocabulary3_1.txt' and 'result_analysis_task3_1.txt') ^u^\n")


def optimize_dic_word_length(dictionary, lessthan, greaterthan):
    remove_list = []
    for item in dictionary:
        if (len(item) <= lessthan) or (len(item) >= greaterthan):
            remove_list.append(item)
    for item in remove_list:
        del(dictionary[item])


task3_2 = input("Please press keyboard to continue Task3.2:")
if task3_2 is not None:
    start_time = time.time()
    print('start3.2..')
    dic3_2 = dictionary.copy()
    optimize_dic_word_length(dic3_2, 2, 9)
    time1 = time.time()
    print("finish saving in dic: ", time1 - start_time, "s")
    result = training_testing_data(dic3_2, 0.5, True, 'wordlength-model.txt', False, 'remove_words3_2.txt', True, 'wordlength-result.txt', True, "vocabulary3_2.txt")
    write_file(str(result), "result_analysis_task3_2.txt")
    time1 = time.time()
    print("finish task3.2: ", time1 - start_time, "s")
    print("Please read 'wordlength-model.txt' ('vocabulary3_2.txt' and 'result_analysis_task3_2.txt') ^u^\n")


def optimize_dic_infrequent_word(dictionary, less, infrequent, type_num):
    remove_list = []
    if less is True:
        for item in dictionary:
            if dictionary.get(item)[type_num] <= infrequent:
                remove_list.append(item)
    else:
        for item in dictionary:
            if dictionary.get(item)[type_num] >= infrequent:
                remove_list.append(item)
    for item in remove_list:
        del (dictionary[item])


def change_infrequent(dic3_3, less, frequent, savevoca, vocafilename):
    start_time = time.time()
    print('start..')
    optimize_dic_infrequent_word(dic3_3, less, frequent, post_type_num)
    time1 = time.time()
    print("finish saving in dic: ", time1 - start_time, "s")
    result = training_testing_data(dic3_3, 0.5, False, 'infrequent-model3_3.txt', False, 'remove_words3_3.txt', False, 'infrequent-result.txt', savevoca, vocafilename)
    print("finish", time1-start_time, "s")
    return result


def cal_frequent(percentage, v_list):
    limit_frequent = 0
    if (v_list is not None) and (len(v_list) > 0):
        index = math.floor(percentage * len(v_list))
        limit_frequent = v_list[index]
    return limit_frequent


def plot_graph(x_lable, x_alias, result_list):
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
        # plt.plot(x, y, 'y.-', x, y, 'ro-', x, y, 'g+-', x, y, 'b^-')
        plt.title('The results in '+types.index[i]+" (different " + x_lable + ")")
        plt.xlabel(x_lable)
        plt.ylabel('%')
        plt.xticks(x, x_alias)
        plt.legend()
    plt.show()

#Task3.3
# frequency = 1, frequency ≤ 5, frequency ≤ 10, frequency ≤ 15 and frequency ≤ 20
# frequency >= 5%, frequency >= 15%, frequency >= 20%, frequency >= 25%
task3_3 = input("Please press keyboard to continue Task3.3:")
if task3_3 is not None:
    start_time = time.time()
    print('start3.3..')
    analyse_fre_result_list = []
    dic3_3 = dictionary.copy()
    x = []
    analyse_fre_result_list.append(change_infrequent(dic3_3, True, 1, True, "vocabulary3_3(1).txt"))
    analyse_fre_result_list.append(change_infrequent(dic3_3, True, 5, True, "vocabulary3_3(5).txt"))
    analyse_fre_result_list.append(change_infrequent(dic3_3, True, 10, True, "vocabulary3_3(10).txt"))
    analyse_fre_result_list.append(change_infrequent(dic3_3, True, 15, True, "vocabulary3_3(15).txt"))
    analyse_fre_result_list.append(change_infrequent(dic3_3, True, 20, True, "vocabulary3_3(20).txt"))
    frequent_list = []
    for item in dictionary:
        frequent_list.append(dictionary.get(item)[post_type_num])
    frequent_list.sort(reverse=True)
    if frequent_list is not None and (len(frequent_list) > 0):
        frequent = cal_frequent(0.05, frequent_list)
        analyse_fre_result_list.append(change_infrequent(dic3_3, False, frequent, True, "vocabulary3_3(top5).txt"))
        frequent = cal_frequent(0.15, frequent_list)
        analyse_fre_result_list.append(change_infrequent(dic3_3, False, frequent, True, "vocabulary3_3(top15).txt"))
        frequent = cal_frequent(0.2, frequent_list)
        analyse_fre_result_list.append(change_infrequent(dic3_3, False, frequent, True, "vocabulary3_3(top20).txt"))
        frequent = cal_frequent(0.25, frequent_list)
        analyse_fre_result_list.append(change_infrequent(dic3_3, False, frequent, True, "vocabulary3_3(top25).txt"))
    x = [1, 5, 10, 15, 20, "top 5%", "top 15%", "top 20%", "top 25%"]
    print("*********\n", str(analyse_fre_result_list))
    plot_graph("frequency", x, analyse_fre_result_list)
    write_file(str(analyse_fre_result_list), "result_analysis_task3_3.txt")
    end_time = time.time()
    print('finishe3.3..', end_time-start_time, "s")
    print("(Please read 'result_analysis_task3_3.txt') ^u^\n")


def change_smooth(smooth, savevoca, vocafilename):
    start_time = time.time()
    print('start smooth: ', smooth)
    result = training_testing_data(dictionary, smooth, False, 'smooth-model.txt', False, 'remove_words3_4.txt', False, 'smooth-result.txt', savevoca, vocafilename)
    time1 = time.time()
    print("finish: ", time1-start_time, "s")
    return result


task3_4 = input("Please press keyboard to continue Task3.4:")
if task3_4 is not None:
    start_time = time.time()
    print('start task 3.4')
    analyse_smooth_result_list = []
    for i in range(11):
        result_list = change_smooth((i+1)/10, True, "vocabulary3_4(" + str((i+1)/10) + ").txt")
        analyse_smooth_result_list.append(result_list)
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plot_graph("smooth", x, analyse_smooth_result_list)
    time1 = time.time()
    print("finish task3.4: ", time1 - start_time, "s")
    write_file(str(analyse_smooth_result_list), "result_analysis_task3_4.txt")
    print("(Please read 'result_analysis_task3_4.txt') ^u^\n")
