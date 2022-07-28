# Goals: Create a program that utilizes childes and mcdi data in order to calculate distributional predictors.
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from cytoolz import itertoolz
import csv
import sys
from scipy.stats.stats import pearsonr


def load_childesdb_data():
    print("Importing CHILDES corpus")
    age_freq_dict = {}

    utype2symbol = {'declarative': ' .',
                    'question': ' ?',
                    'imperative_emphatic': ' ?'}

    f_handle = open('/Volumes/Teal SSD/Research/Pro-KWo/|1|select_childes.csv', 'r', encoding="utf8" ) 
    row_dicts = csv.DictReader(f_handle)

    childesdb_data = []
    speaker_dict = {}


    row_counter = 0
    for row_dict in row_dicts:

        # create a dictionary of all the speakers we have encountered
        if not row_dict['speaker_code'] in speaker_dict:
            speaker_dict[row_dict['speaker_code']] = 1  # if item is not found add an entry to the dictionary.
            speaker_dict[row_dict['speaker_code']] += 1  # If speaker code is already in the dictionary increase it by1.

        # creating the boundary symbols
        try:
            boundary_symbol = utype2symbol[row_dict['type']]
        except KeyError:
            boundary_symbol = ' .'  # if type !=  any of the items in the dictionary utype2symbol replace with ' .'
        utterance = row_dict['gloss'] + boundary_symbol  # adds punctuation to the 'gloss' utterance

        transcript_id = row_dict['transcript_id']
        speaker_code = row_dict['speaker_code']
        age = row_dict['target_child_age']
        if age != 'NA':
            target_age = int(float(age))
            token_list = utterance.split()
            if target_age < 16:
                target_age = 16
            sample = [transcript_id, speaker_code, target_age, token_list]

            if target_age <= 30:
                childesdb_data.append(sample)
                if target_age in age_freq_dict:
                    age_freq_dict[target_age] += 1
                else:
                    age_freq_dict[target_age] = 1

        row_counter += 1
        if row_counter % 20000 == 0:
            print("     Finished importing {} rows".format(row_counter))

    return childesdb_data  # list of samples e.g. [speaker, age, w1, w2, ... punctuation]



def load_mcdi_data():
    print('Loading MCDI')
    mcdi_data = pd.read_csv('/Volumes/Teal SSD/Research/Pro-KWo/|2|preprocessed_mcdi.csv')
    mcdi_data = pd.DataFrame(mcdi_data)
    target_words = pd.read_csv("/Volumes/Teal SSD/Research/Pro-KWo/|2|target_words.csv")
    return mcdi_data, target_words


def create_age_data_structures(mcdi_data):
    print('Creating Age Data Structures')
    age_list = []
    age_index_dict = {}
    age_column = mcdi_data['Age']
    age_column = pd.Series(age_column).unique()
    for row in age_column:
        age_list.append(row)
    counter = 0
    for i in range(len(age_list)):
        age_index_dict[age_list[i]] = counter
        counter += 1
    return age_list, age_index_dict


def create_target_data_structures(target_words):
    print('Creating Target Data Structures')
    target_list = []
    target_index_dict = {}

    target_words_column = target_words['Word']

    for row in target_words_column:
        target_list.append(row)
        target_list.sort()

    counter = 0

    for i in range(len(target_list)):
        target_list[i] = target_list[i].lower()
        target_index_dict[target_list[i].lower()] = counter

        counter += 1

    return target_list, target_index_dict



def create_doc_data_structures(childesdb_data):
    print('Creating Document Data Structures')
    doc_list = []
    doc_index_dict = {}
    doc_counter = 0

    for i in range(len(childesdb_data)):
        doc_id = childesdb_data[i][0]
        if doc_id not in doc_index_dict:
            doc_list.append(doc_id)
            doc_index_dict[doc_id] = doc_counter
            doc_counter += 1

    return doc_list, doc_index_dict


def create_word_data_structures(childesdb_data):
    print('Creating Word Data Structures')
    word_list = []
    word_index_dict = {}
    word_counter = 0

    for i in range(len(childesdb_data)):
        utterance = childesdb_data[i][3]
        for token in utterance:
            token = token.lower()
            if token not in word_index_dict:
                word_list.append(token)
                word_index_dict[token] = word_counter
                word_counter += 1

    return word_list, word_index_dict

def target_frequency_age_matrix(target_index_dict, age_index_dict, childesdb_data):
    print('Creating Word Frequences Matrices by Age')
# Print 20 most freq words
    num_targets = len(target_index_dict)
    num_ages = len(age_index_dict)

    target_age_freq_matrix = np.zeros([num_targets, num_ages], int)
    cumulative_target_frequency_matrix = np.zeros([num_targets, num_ages], int)

    age_corp_size_array = np.zeros([len(age_index_dict)], int)

    print(age_index_dict)

    for i in range(len(childesdb_data)):
        utterance = childesdb_data[i][3]
        age = childesdb_data[i][2]

        for token in utterance:
            token = token.lower()

            age_index = age_index_dict[age]
            if token in target_index_dict:
                target_age_freq_matrix[target_index_dict[token], age_index] += 1
            age_corp_size_array[age_index] += 1

    for i in range(num_targets):
        for j in range(num_ages):
            if j == 0:
                cumulative_target_frequency_matrix[i,j] = target_age_freq_matrix[i,j]
            else:
                cumulative_target_frequency_matrix[i,j] = target_age_freq_matrix[i,j] + cumulative_target_frequency_matrix[i,j-1]

    return target_age_freq_matrix, cumulative_target_frequency_matrix, age_corp_size_array


def co_occurence_matrix(target_index_dict, age_index_dict, childesdb_data):

    print('Creating co-occurrence matrices')
    window_type = 'forward' # forward, backward, summed, concatenated
    window_size = 7
    window_weight = 'flat' # linear or flat
    PAD = '*PAD*'

    # The goal is to create a 3 dimensional array of the following x,y,z dimensions: MCDI words X MCDI words X Age
    num_targets = len(target_index_dict)
    num_ages = len(age_index_dict)
    cooc_matrix_by_age_list = []
    cumulative_cooc_matrix_by_age_list = []

    corpus_by_age_list = []
    for i in range(num_ages):
        corpus_by_age_list.append([])

    # Then specify what items (words) will be updating the correct row and columns.
    for i in range(len(childesdb_data)):
        utterance = childesdb_data[i][3]
        age = childesdb_data[i][2]
        age_index = age_index_dict[age]
        corpus_by_age_list[age_index] += utterance

    # now we are ready to start counting co-occurrences for each age

    for i in range(num_ages):
        print("     Age Index: {}".format(i))
        cooc_matrix = np.zeros ( [ num_targets , num_targets ] , float )
        cumulative_cooc_matrix = np.zeros ( [ num_targets , num_targets ] , float )
        current_corpus = corpus_by_age_list[i]
        if len(current_corpus) > 0:
            current_corpus += [PAD] * window_size  # add pad such that all co-occurrences in last window are captured
            windows = itertoolz.sliding_window(window_size, current_corpus)

            for w in windows:
                for word1, word2, dist in zip([w[0]] * (window_size - 1), w[1:], range(1, window_size)):
                    # increment
                    if word1 == PAD or word2 == PAD:
                        continue

                    if word1 not in target_index_dict:
                        continue

                    if word2 not in target_index_dict:
                        continue
                    word1_index = target_index_dict[word1]
                    word2_index = target_index_dict[word2]

                    if window_weight == "linear":
                        cooc_matrix[word1_index, word2_index] += window_size - dist
                    elif window_weight == "flat":
                        cooc_matrix[word1_index, word2_index] += 1
            # window_type
            if window_type == 'forward':
                final_matrix = cooc_matrix
            elif window_type == 'backward':
                final_matrix = cooc_matrix.transpose()
            elif window_type == 'summed':
                final_matrix = cooc_matrix + cooc_matrix.transpose()
            elif window_type == 'concatenate':
                final_matrix = np.concatenate((cooc_matrix, cooc_matrix.transpose()))
            else:
                raise AttributeError('Invalid arg to "window_type".')

            cooc_matrix_by_age_list.append(final_matrix)

    current_cumul_cooc_matrix = np.zeros([num_targets, num_targets], float)

    for i in range(num_ages):
        current_cooc_matrix = cooc_matrix_by_age_list[i]
        current_cumul_cooc_matrix += current_cooc_matrix
        cumulative_cooc_matrix_by_age_list.append(current_cumul_cooc_matrix.copy())

    return cooc_matrix_by_age_list, cumulative_cooc_matrix_by_age_list

def target_lexical_diversity_by_age(age_list, target_list, cumulative_cooc_matrix_by_age_list,target_index_dict,age_index_dict):
    print('Calculating Lexical Diversity')
    num_targets = len(target_index_dict)
    num_ages = len(age_index_dict)

    ld_age_matrix = np.zeros([num_targets, num_ages], float)

    for i in range(num_ages):
        for j in range(num_targets):
            current_target_coocs = cumulative_cooc_matrix_by_age_list[i][j, :]
            num_nonzero = np.count_nonzero(current_target_coocs)
            prop_nonzero = num_nonzero / num_targets
            ld_age_matrix[j, i] = prop_nonzero

    return ld_age_matrix

def dd_matrix(target_index_dict, age_index_dict, doc_index_dict, doc_list, childesdb_data):
    num_targets = len(target_index_dict)
    num_ages = len(age_index_dict)
    num_docs = len(doc_index_dict)

    docs_per_age_matrix = np.zeros([num_ages], float)
    dd_by_age_matrix = np.zeros([num_targets, num_ages], float)
    dd_by_age_matrix_cumul = np.zeros([num_targets, num_ages], float)

    target_document_freq_matrix = np.zeros([num_targets, num_docs], float)
    doc_age_dict = {}
    encountered_doc_index_dict = {}

    for i in range(len(childesdb_data)):
        document_id = childesdb_data[i][0]
        age = childesdb_data[i][2]
        utterance = childesdb_data[i][3]

        age_index = age_index_dict[age]

        doc_age_dict[document_id] = age
        if document_id not in encountered_doc_index_dict:
            docs_per_age_matrix[age_index] += 1

        for token in utterance:
            if token in target_index_dict:
                if document_id in doc_index_dict:
                    target_index = target_index_dict[token]
                    document_index = doc_index_dict[document_id]
                    target_document_freq_matrix[target_index, document_index] += 1

    cumul_docs_per_age_matrix = np.zeros([num_ages], float)
    cumul_docs_per_age_matrix[0] = docs_per_age_matrix[0]
    for i in range(num_ages - 1):
        cumul_docs_per_age_matrix[i+1] = cumul_docs_per_age_matrix[i] + docs_per_age_matrix[i+1]

    nonzero_sum_matrix = np.zeros([num_targets, num_ages], float)

    # 19 20 21 22 23
    # 0   1  2  3  4
    #  current age = 21
    # current age index = 2
    # increment 2, 3, 4

    for i in range(num_targets):
        for j in range(num_docs):
            if target_document_freq_matrix[i, j] > 0:
                document_id = doc_list[j]
                current_age = doc_age_dict[document_id]
                current_age_index = age_index_dict[current_age]

                increment_window_size = num_ages - current_age_index + 1
                for k in range(increment_window_size):
                    try:
                        nonzero_sum_matrix[i, current_age_index + k] += 1
                    except IndexError:
                        print("outofbounds")

    for i in range(num_targets):
        for j in range(num_ages):
            dd_by_age_matrix[i,j] = nonzero_sum_matrix[i,j] / num_docs

    for i in range(num_targets):
        for j in range(num_ages):
            if j == 0:
                dd_by_age_matrix_cumul[i, j] = dd_by_age_matrix[i, j]
            else:
                dd_by_age_matrix_cumul[i, j] = dd_by_age_matrix[i, j] + dd_by_age_matrix_cumul[i, j - 1]

    return dd_by_age_matrix, cumul_docs_per_age_matrix, nonzero_sum_matrix


def get_mcdip_by_age_matrix(age_index_dict, target_list, target_word_index_dict, mcdi_data):
    #pro.write ( 'ProKWo\n' )
    mcdip = pd.DataFrame(mcdi_data)

    num_ages = len(age_index_dict)
    num_targets = len(target_word_index_dict)

    # convert mcdip in the dataframe, to an age (rows) x mcdi_word (cols) matrix
    mcdip_matrix = np.zeros([num_ages, num_targets], float)
    df_data = mcdi_data.values
    # Create an empty matrix and populate with corresponding mcdip scores.
    for i in range(len(df_data)):
        age = df_data[i][2]
        target = df_data[i][3].lower()
        mcdip = df_data[i][7]
        age_index = age_index_dict[age]
        target_index = target_word_index_dict[target]
        mcdip_matrix[age_index, target_index] = mcdip

    return mcdip_matrix

def calculate_kw_cooc(mcdip_matrix, cumulative_cooc_matrix_by_age_list):
    # loads in mcdi_data
    # convert cooc-matrix from a list into a numpy.array
    # cooc_matrix = np.asanyarray(cooc_matrix_by_age_list)
    # returns the following 3 dimensional array (15 ages * 629 target words * 629 target words)

    # create a an empty list to be populated by KWCOOC values

    num_ages = mcdip_matrix.shape[0]
    num_targets = mcdip_matrix.shape[1]

    kwcooc_counts = np.zeros([num_ages, num_targets], float)
    kwcooc_proportions = np.zeros([num_ages, num_targets], float)

    mcdip_values = np.copy(mcdip_matrix)
    # if shuffle: 
    #     np.random.shuffle(np.transpose(mcdip_values))

    for i in range(num_ages):
        cooc_matrix_for_current_age = cumulative_cooc_matrix_by_age_list[i]

        for j in range(num_targets):
            cooc_row = cooc_matrix_for_current_age[j,:]
            mcdip_row = mcdip_values[i, :]
            kwcooc_counts[i,j] = np.dot(cooc_row, mcdip_row)
            cooc_sum = cooc_row.sum()
            if cooc_sum > 0:
                kwcooc_proportions[i, j] = kwcooc_counts[i,j] / cooc_sum
            else:
                kwcooc_proportions[i, j] = 0

    kwcooc_proportions_transposed= kwcooc_proportions.transpose()


    #pro.write ( '{:.5f}\n'.format ( kwcooc_proportions) )
    #np.savetxt("ProKWo.csv", kwcooc_proportions, delimiter=",", fmt = "%10s")
    # np.save("ProKWoze", kwcooc_proportions)
    # df = pd.DataFrame ( kwcooc_proportions )
    # df.to_csv ( "ProKWo_Cumul.csv" , index=False )

    return kwcooc_proportions,kwcooc_proportions_transposed


def output_cooc_matrices (age_list, target_list, cooc_matrix_by_age_list) :
    print('Outputting Co-occurrence Matrices')
    f = open('cooc_matrices.txt', 'w')
    for i in range(len(age_list)):
        for j in range(len(target_list)):
            f.write('{} {}'.format(age_list[i], target_list[j]))
            for k in range(len(target_list)):
                cooc = cooc_matrix_by_age_list[i][j, k]
                f.write(' {}'.format(cooc))
            f.write('\n')
    f.close()

def compute_mcdip_correlation(predictor, mcdip_matrix):
    num_ages = mcdip_matrix.shape[0]
    correlation_list = []
    for i in range(num_ages):
        current_age_mcdip = mcdip_matrix[i,:]
        current_age_predictor = predictor[i, :]
        correlation = pearsonr(current_age_mcdip, current_age_predictor)
        correlation_list.append(correlation[0])
    return correlation_list

def calculate_prokwo_shuffle_correlations(mcdip_matrix, cumulative_cooc_matrix_by_age_list, num_shuffles):

    num_ages = mcdip_matrix.shape[0]
    num_targets = mcdip_matrix.shape[1]

    prokwo_shuffled_correlation_matrix = np.zeros([num_ages, num_shuffles], float)

    for i in range(num_shuffles):
        prokwo_shuffled_matrix, prokwo_shuffled_matrix_transposed=calculate_kw_cooc(mcdip_matrix, 
                                                                    cumulative_cooc_matrix_by_age_list,
                                                                    True)

        correlation_list = compute_mcdip_correlation(prokwo_shuffled_matrix, mcdip_matrix)
        prokwo_shuffled_correlation_matrix[:,i] = np.array(correlation_list)

        if i % 100 == 0:
            print("     Finished {} shuffles".format(i))

    mean_correlations = prokwo_shuffled_correlation_matrix.mean(1)
    return mean_correlations


def output_target_data(target_list,
                       target_index_dict,
                       age_list,
                       age_index_dict,
                       mcdi_data,
                       target_age_freq_matrix,
                       cumulative_target_frequency_matrix,
                       target_ld_by_age_matrix,
                        cumul_docs_per_age_matrix,
                        kwcooc_proportions_transposed
                       ):
    print('Outputting Target Data')
    #MCDIp = []
    num_targets = len(target_list)
    num_ages = len(age_list)
    f = open('|3|childes_original.csv', 'w')
    f.write('Age,Word,C_freq,ld,dd,pkc\n')
    for i in range(num_targets):
        for j in range(num_ages):
            age = age_list[j]
            target = target_list[i]
            freq = target_age_freq_matrix[i,j]
            cumulative_freq = cumulative_target_frequency_matrix[i,j]
            #bool_ind = (mcdi_data['Age'] == age) & (mcdi_data['Word'] == target)
            ld = target_ld_by_age_matrix[i,j]
            dd=  cumul_docs_per_age_matrix[i,j]
            pkc=kwcooc_proportions_transposed[i,j]
            print(age,target,freq,cumulative_freq,ld,dd,pkc)

            # try:
            #     MCDIp = mcdi_data[bool_ind]['MCDIp'].values[0]
            # except IndexError:
            #     print('Did not find', target)
            # print(age, target, freq, MCDIp)
            f.write('{},{},{},{:.9f}, {:.9f},{:.9f}\n'.format(age, target,cumulative_freq,ld, dd,pkc))


def main():
    # childes_file = sys.argv[1]
    # mcdi_file = sys.argv[2]

    childesdb_data = load_childesdb_data()
    mcdi_data, target_words = load_mcdi_data()
    age_list, age_index_dict = create_age_data_structures(mcdi_data)
    target_list, target_index_dict = create_target_data_structures(target_words)

    doc_list, doc_index_dict = create_doc_data_structures(childesdb_data)

    word_list, word_index_dict = create_word_data_structures(childesdb_data)
    target_age_freq_matrix, cumulative_target_frequency_matrix, age_corp_size_array = target_frequency_age_matrix(target_index_dict, age_index_dict, childesdb_data)
    


    cooc_matrix_by_age_list, cumulative_cooc_matrix_by_age_list = co_occurence_matrix(target_index_dict, age_index_dict, childesdb_data)
 
    # for j in range(len(age_list)):
    #     print("Age {}".format(age_list[j]))
    # for i in range(len(target_list)):
    #     print(target_list[i], cooc_matrix_by_age_list[-1][i,:])

    dd_by_age_matrix, nonzero_sum_matrix, dd_by_age_matrix_cumul = dd_matrix(target_index_dict, age_index_dict, doc_index_dict, doc_list, childesdb_data)
    target_ld_by_age_matrix = target_lexical_diversity_by_age(age_list, target_list, cumulative_cooc_matrix_by_age_list,target_index_dict,age_index_dict)
    output_cooc_matrices(age_list, target_list, cooc_matrix_by_age_list)


    # for i in range(len(target_list)):
    #     print(target_list[i], dd_by_age_matrix_cumul[i,:])

    mcdip_matrix = get_mcdip_by_age_matrix(age_index_dict, target_list, target_index_dict, mcdi_data)

    prokwo_matrix, prokwo_matrix_transposed=calculate_kw_cooc(mcdip_matrix, 
                                                              cumulative_cooc_matrix_by_age_list)

    #correlation_list = compute_mcdip_correlation(prokwo_matrix, mcdip_matrix)
    #print(correlation_list)

    #shuffle_correlation_list = calculate_prokwo_shuffle_correlations(mcdip_matrix, cumulative_cooc_matrix_by_age_list, 1000)
    #print(shuffle_correlation_list)




    # print(age_corp_size_array)

    output_target_data(target_list,
                    target_index_dict,
                    age_list,
                    age_index_dict,
                    mcdi_data,
                    target_age_freq_matrix,
                    cumulative_target_frequency_matrix,
                    target_ld_by_age_matrix,
                    dd_by_age_matrix,
                    prokwo_matrix_transposed
                    )

main()
