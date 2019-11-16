import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
import math
from decimal import Decimal

lemmatizer = WordNetLemmatizer()
# read file and get data
# df = pd.read_csv('hn2018_2019.csv', usecols=['Title', 'Post Type', 'Created At']) #, encoding='latin-1')
df = pd.read_csv('temp1.csv', usecols=['Title', 'Post Type', 'Created At']) #, encoding='latin-1')
df18 = df[(True ^ df['Created At'].str.contains('2019'))]
df19 = df[(True ^ df['Created At'].str.contains('2018'))]
list_dt18 = np.array(df18)
list_dt19 = np.array(df19)


# count post_type and the number of post_type
df18['Post Type'].value_counts()
types = df18['Post Type'].value_counts()
type_dict = {}  # {post_type name:[index, total_num], ...}
type_dict1 = {}  # {index:post_type name, ...}

post_type_num = len(types)
for i in range(post_type_num):
    type_dict[types.index[i]] = [i, types[i]]
    type_dict1[i] = types.index[i]

print("finish reading")


punctuation = ["!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", ":", ";",
              "<", "=", ">", "?", "@", "[", "\\", "]", "^", "`", "{", "|", "}", "~", " ",
              "/", ".", "-", "_"]

symbol_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ".", "+", "-", "*", "/"]

letter_table = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
              , '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
remove_set = set()


#  split ' from 'word and become ["'","word"]
def split_word(word_list):
    new_word_list = []
    for item in word_list:
        item = item.strip()
        if item.startswith("'") and (len(item) > 1):
            new_word_list.append("'")
            new_word_list.append(item[1:])
        elif item is not "":
            new_word_list.append(item)
    return new_word_list


def split_cd(word):
    split_word = []
    start = 0
    for index0 in range(len(word)):
        start_character = word[index0]
        if start_character in symbol_number:
            start = index0 + 1
        else:
            break
    split_word.append(word[0:start])
    split_word.append(word[start:])
    return split_word


def recreate_list(word_list, save_in_remove_file):
    new_list = []
    prior_is_noun = False
    for index in range(len(word_list)):  # one tuple: (word, pos)
        if prior_is_noun is False:
            item = word_list[index]
            if item[1] in "CD":
                number_word = split_cd(item[0])
                if (number_word[0] is not "") and save_in_remove_file:
                    remove_set.add(number_word[0])
                if number_word[1] is not "":
                    new_list.append(number_word[1])
            elif item[1].startswith('J'):  # adj
                if item[1] is "JJ":
                    ordinal_numeral = True
                    for c in item[0]:
                        if c in symbol_number:
                            continue
                        else:
                            ordinal_numeral = False
                            break
                    if ordinal_numeral is True and save_in_remove_file:
                        remove_set.add(item[0])
                    else:
                        token = lemmatizer.lemmatize(item[0], 'a')
                        new_list.append(token)
                else:
                    token = lemmatizer.lemmatize(item[0], 'a')
                    new_list.append(token)
            elif item[1].startswith('R'):  # adv
                token = lemmatizer.lemmatize(item[0], 'r')
                new_list.append(token)
            elif item[1].startswith('V'):  # verb
                token = lemmatizer.lemmatize(item[0], 'v')
                new_list.append(token)
            elif item[1].startswith('N'):  # noun
                token = lemmatizer.lemmatize(item[0], 'n')
                if (index + 1) < len(word_list):
                    item1 = word_list[index+1]
                    if item1[1].startswith('N'):
                        token1 = lemmatizer.lemmatize(item1[0], 'n')
                        new_list.append(token + " " + token1)
                        prior_is_noun = True
                    else:
                        new_list.append(token)
                else:  # last one
                    new_list.append(token)
            else:
                new_list.append(item[0])
        elif prior_is_noun is True:
            prior_is_noun = False
    return new_list


def filter_title(title, removefile):
    words = word_tokenize(title)
    words1 = split_word(words)
    words_list = nltk.pos_tag(words1)
    # print(words_list)
    words_list1 = recreate_list(words_list, removefile)
    return words_list1


# create dictionary
def create_dictionary(removefile):
    vdict = {}  # {key(word), [num of type1, num of type2, num of type3, num of type4,... , total num]}
    i = 0
    for obj in list_dt18:
        title = obj[0].lower()
        post_type = obj[1]
        post_type_index = type_dict[post_type][0]  # {key, [index, total_num], ...}
        words_list = filter_title(title, removefile)
        i = i+1
        print(i, words_list)
        for token in words_list:
            if vdict.get(token) is None:
                vdict[token] = [0] * (post_type_num + 1)
            list1 = vdict.get(token)
            list1[post_type_index] = list1[post_type_index] + 1
            list1[post_type_num] = list1[post_type_num] + 1
            vdict[token] = list1
    return vdict


def create_vocabulary(vdict):
    vocabulary = sorted(vdict.keys())
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
    log_prob = math.log(probability, 10)
    log_prob = Decimal(log_prob).quantize(Decimal("0.0000000000"))
    return log_prob


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
        # print(a_token)
        s = s + str(row_num) + "  " + str(token)
        list_value = []
        for index in range(post_type_num):
            word_prob = cal_prob(vdict.get(token)[index], smooth, types_value_list[index], vocabulary_len)
            if savefile:
                s = s + "  " + str(vdict.get(token)[index]) + "  " + str(word_prob)
            list_value.append(word_prob)
        vdict_with_pro[token] = list_value
        s = s + "\r\n"
        row_num = row_num + 1
    if savefile:
        output_file18 = open(filename, 'w')
        output_file18.write(s)
        output_file18.close()
    return vdict_with_pro


def training_data(dictionary, smooth, savefile, filename, removefile, remove_filename):
    vocabulary = create_vocabulary(dictionary)
    types_value_list = sum_type_value(dictionary)
    vdict_with_pro = get_probability(smooth, dictionary, vocabulary, types_value_list, savefile, filename)
    if removefile:
        remove_word = ""
        for item in remove_set:
            remove_word = remove_word + item + "\n"
        output_file = open(remove_filename, 'w')
        output_file.write(remove_word)
        output_file.close()
    return vdict_with_pro


# Task1
start_time = time.time()
print('start..')
dictionary = create_dictionary(True)
time1 = time.time()
print("finish saving in dic: ", time1 - start_time, "s")

vdict_with_pro = training_data(dictionary, 0.5, True, 'model-2018.txt', True, 'remove_words.txt')
time1 = time.time()
print("finish pro: ", time1 - start_time, "s")


# Task2
def cal_score(tittle_word_list, total_doc_len, vdict_with_pro):
    words_prob = [0] * post_type_num
    for index in type_dict1:
        words_prob[index] = math.log(type_dict.get(type_dict1.get(index))[1] / total_doc_len, 10)
        words_prob[index] = float(Decimal(words_prob[index]).quantize(Decimal("0.0000000000")))
    for token in tittle_word_list:
        if vdict_with_pro.get(token) is not None:
            for index in type_dict1:  # {index:key(type), ...}
                token_prob = vdict_with_pro.get(token)  # {key, [log, log...]..}
                words_prob[index] = words_prob[index] + float(token_prob[index])
        else:
            continue
    return words_prob


def predict_type(prob):
    index = prob.index(max(prob))
    return type_dict1.get(index)


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
        words_list = filter_title(title.lower(), False)
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
        output_file = open(filename, 'w')
        output_file.write(s)
        output_file.close()

    predict_list = np.sum(confusion_matrix, axis=0)
    real_list = np.sum(confusion_matrix, axis=1)
    total_number = np.sum(real_list)

    matrix = "confusion matrix/contingency table:\nreal\\predict"
    for i in range(post_type_num):
        matrix = matrix + "\t" + types.index[i]
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
    for index in range(post_type_num):
        right = right + confusion_matrix[index][index]
    precision = [0] * post_type_num
    recall = [0] * post_type_num
    f1_measure = [0] * post_type_num
    accuracy = right / total_number
    for index in range(post_type_num):
        precision[index] = confusion_matrix[index][index] / predict_list[index]
        recall[index] = confusion_matrix[index][index] / real_list[index]
        f1_measure[index] = (2 * precision[index] * recall[index]) / (precision[index] + recall[index])

    print("Accuracy", accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("F1_Measure", f1_measure)
    return[accuracy, precision, recall, f1_measure]

# Task2
task2 = input("Please press keyboard to continue Task2:")
if task2 is not None:
    testing_data(vdict_with_pro, True, 'baseline-result.txt')
time1 = time.time()
print("finish task2: ", time1-start_time, "s")


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

    return dictionary


task3_1 = input("Please press keyboard to continue Task3.1:")
if task3_1 is not None:
    start_time = time.time()
    print('start3.1..')
    dic3_1 = optimize_dic_stop_word(dictionary)
    time1 = time.time()
    print("finish saving in dic: ", time1 - start_time, "s")
    vdict_with_pro = training_data(dic3_1, 0.5, True, 'stopword-model.txt', False, 'remove_words3_1.txt')
    time1 = time.time()
    print("finish pro3.1: ", time1 - start_time, "s")
    # Test
    print("start testing data...")
    testing_data(vdict_with_pro, True, 'stopword-result.txt')
    time1 = time.time()
    print("finish task3.1: ", time1-start_time, "s")


def optimize_dic_word_length(dictionary, lessthan, greaterthan):
    remove_list = []
    for item in dictionary:
        if (len(item) <= lessthan) or (len(item) >= greaterthan):
            remove_list.append(item)
    for item in remove_list:
        del(dictionary[item])
    return dictionary


task3_2 = input("Please press keyboard to continue Task3.2:")
if task3_2 is not None:
    start_time = time.time()
    print('start3.2..')
    dic3_2 = optimize_dic_word_length(dictionary, 2, 9)
    time1 = time.time()
    print("finish saving in dic: ", time1 - start_time, "s")
    vdict_with_pro = training_data(dic3_2, 0.5, True, 'wordlength-model.txt', False, 'remove_words3_2.txt')
    time1 = time.time()
    print("finish pro3.2: ", time1 - start_time, "s")
    # Test
    print("start testing data...")
    testing_data(vdict_with_pro, True, 'wordlength-result.txt')
    time1 = time.time()
    print("finish task3.2: ", time1-start_time, "s")


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
    return dictionary


def change_infrequent(dictionary, less, frequent):
    start_time = time.time()
    print('start3.3..')
    dic3_3 = optimize_dic_infrequent_word(dictionary, less, frequent, post_type_num)
    time1 = time.time()
    print("finish saving in dic: ", time1 - start_time, "s")
    vdict_with_pro = training_data(dic3_3, 0.5, False, 'infrequent-model3_3.txt', False, 'remove_words3_3.txt')
    time1 = time.time()
    print("finish pro3.1: ", time1 - start_time, "s")
    # Test
    print("start testing data...")
    # get [accuracy, precision, recall, f1_measure]
    result_list = testing_data(vdict_with_pro, False, 'infrequent-result.txt')
    time1 = time.time()
    print("finish task3.3: ", time1-start_time, "s")
    return [dic3_3, result_list]


def cal_frequent(percentage, v_list):
    if v_list is not None and len(v_list)>0:
        index = math.floor(percentage * len(v_list))
        print(index, v_list)
        limit_frequent = v_list[index]
    return limit_frequent


#Task3.3
# frequency = 1, frequency ≤ 5, frequency ≤ 10, frequency ≤ 15 and frequency ≤ 20
# frequency >= 5%, frequency >= 15%, frequency >= 20%, frequency >= 25%
task3_3 = input("Please press keyboard to continue Task3.3:")
if task3_3 is not None:
    analyse_fre_result_list = []
    new_dictionary = change_infrequent(dictionary, True, 1)
    analyse_fre_result_list.append(new_dictionary[1])
    new_dictionary = change_infrequent(new_dictionary[0], True, 5)
    analyse_fre_result_list.append(new_dictionary[1])
    new_dictionary = change_infrequent(new_dictionary[0], True, 10)
    analyse_fre_result_list.append(new_dictionary[1])
    new_dictionary = change_infrequent(new_dictionary[0], True, 15)
    analyse_fre_result_list.append(new_dictionary[1])
    new_dictionary = change_infrequent(new_dictionary[0], True, 20)
    analyse_fre_result_list.append(new_dictionary[1])
    frequent_list = []
    for item in dictionary:
        frequent_list.append(dictionary.get(item)[post_type_num])
    frequent_list.sort(reverse=True)
    if frequent_list is not None and len(frequent_list)>0 :
        frequent = cal_frequent(0.05, frequent_list)
        new_dictionary1 = change_infrequent(dictionary, False, frequent)
        analyse_fre_result_list.append(new_dictionary1[1])
        frequent = cal_frequent(0.15, frequent_list)
        new_dictionary1 = change_infrequent(new_dictionary1[0], False, frequent)
        analyse_fre_result_list.append(new_dictionary1[1])
        frequent = cal_frequent(0.2, frequent_list)
        new_dictionary1 = change_infrequent(new_dictionary1[0], False, frequent)
        analyse_fre_result_list.append(new_dictionary1[1])
        frequent = cal_frequent(0.25, frequent_list)
        new_dictionary1 = change_infrequent(new_dictionary1[0], False, frequent)
        analyse_fre_result_list.append(new_dictionary1[1])
    else:
        print("")



def change_smooth(smooth):
    start_time = time.time()
    print('start3.4..', smooth)
    vdict_with_pro = training_data(dictionary, smooth, False, 'smooth-model.txt', False, 'remove_words3_4.txt')
    time1 = time.time()
    print("finish pro3.4: ", time1 - start_time, "s")
    # Test
    print("start testing...")
    result_list = testing_data(vdict_with_pro, False, 'smooth-result.txt')
    time1 = time.time()
    print("finish task3.4: ", time1-start_time, "s")
    return result_list

task3_4 = input("Please press keyboard to continue Task3.3:")
if task3_4 is not None:
    analyse_smooth_result_list = []
    for i in range(11):
        result_list = change_smooth(i/10)
        analyse_smooth_result_list.append(result_list)

