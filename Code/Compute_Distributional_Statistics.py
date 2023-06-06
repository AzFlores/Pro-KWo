# Goals: Create a program that utilizes childes and mcdi data in order to calculate distributional predictors.
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from cytoolz import itertoolz
import csv
import sys
import os
from scipy.stats.stats import pearsonr

def load_childesdb_data(file_path):
    """Load CHILDES corpus data from the specified file path.
    
    Args:
        file_path (str): The path to the file containing CHILDES corpus data.
    
    Returns:
        list: A list of lists, each containing a sample from the CHILDES corpus data.
    """
    print("Importing CHILDES corpus")

    age_freq_dict = {}
    childesdb_data = []
    speaker_dict = {}
    row_counter = 0

    utype2symbol = {
        'declarative': ' .',
        'question': ' ?',
        'imperative_emphatic': ' ?'
    }

    with open(file_path, 'r', encoding="utf8") as f_handle:
        row_dicts = csv.DictReader(f_handle)

        for row_dict in row_dicts:
            speaker_code = row_dict['speaker_code']
            speaker_dict.setdefault(speaker_code, 0)
            speaker_dict[speaker_code] += 1

            boundary_symbol = utype2symbol.get(row_dict['type'], ' .')
            utterance = row_dict['gloss'] + boundary_symbol

            transcript_id = row_dict['transcript_id']
            age = row_dict['target_child_age']

            if age not in ('NA', ''):
                target_age = max(16, int(float(age)))

                # Only process the data if the target age is less than or equal to 30
                if target_age <= 30:
                    token_list = utterance.split()

                    sample = [transcript_id, speaker_code, target_age, token_list]
                    childesdb_data.append(sample)

                    age_freq_dict.setdefault(target_age, 0)
                    age_freq_dict[target_age] += 1

            row_counter += 1
            if row_counter % 20000 == 0:
                print(f"     Finished importing {row_counter} rows")

    return childesdb_data



def load_mcdi_data(mcdi_file_path, target_words_file_path):
    """Load MCDI data and target words data from the specified file paths.
    
    Args:
        mcdi_file_path (str): The path to the file containing MCDI data.
        target_words_file_path (str): The path to the file containing target words data.
    
    Returns:
        tuple: A tuple containing two DataFrames, one for MCDI data and another for target words data.
    """
    print('Loading MCDI')
    mcdi_data = pd.read_csv(mcdi_file_path)
    target_words_data = pd.read_csv(target_words_file_path)
    
    return mcdi_data, target_words_data


def create_age_data_structures(mcdi_data):
    """Create data structures for age data.
    
    Args:
        mcdi_data (DataFrame): A DataFrame containing MCDI data.
    
    Returns:
        tuple: A tuple containing a list of unique ages and a dictionary mapping ages to indices.
    """
    print('Creating Age Data Structures')
    unique_ages = mcdi_data['age'].unique().tolist()
    age_to_index = {age: idx for idx, age in enumerate(unique_ages)}
    
    return unique_ages, age_to_index



def create_target_data_structures(target_words):
    """Create data structures for target words data.
    Args: 
        target_words (DataFrame): A DataFrame containing target words data.
    Returns: 
        tuple: A tuple containing a list of target words and a dictionary mapping target words to indices.
"""
    print('Creating Target Data Structures')
    target_list = sorted(target_words['word'].str.lower().tolist())

    target_index_dict = {word: idx for idx, word in enumerate(target_list)}
    return target_list, target_index_dict



def create_doc_data_structures(childes_corpus_data):
    """Create data structures for document data.
    
    Args:
        childes_corpus_data (list): A list containing CHILDES corpus data.
    
    Returns:
        tuple: A tuple containing a list of document IDs and a dictionary mapping document IDs to indices.
    """
    print('Creating Document Data Structures')
    doc_ids = []
    doc_id_to_index = {}

    for doc_id, _, _, _ in childes_corpus_data:
        if doc_id not in doc_id_to_index:
            doc_ids.append(doc_id)
            doc_id_to_index[doc_id] = len(doc_ids) - 1

    return doc_ids, doc_id_to_index


def create_word_data_structures(childes_corpus_data):
    """Create data structures for word data.
    
    Args:
        childes_corpus_data (list): A list containing CHILDES corpus data.
    
    Returns:
        tuple: A tuple containing a list of words and a dictionary mapping words to indices.
    """
    print('Creating Word Data Structures')
    words = []
    word_to_index = {}

    for _, _, _, utterance in childes_corpus_data:
        for token in utterance:
            token = token.lower()
            if token not in word_to_index:
                words.append(token)
                word_to_index[token] = len(words) - 1

    return words, word_to_index


def target_frequency_age_matrix(target_to_index, age_to_index, childes_corpus_data):
    """Create word frequencies matrices by age.
    Args:
        target_to_index (dict): A dictionary mapping target words to indices.
        age_to_index (dict): A dictionary mapping ages to indices.
        childes_corpus_data (list): A list containing CHILDES corpus data.

    Returns:
        tuple: A tuple containing target_age_freq_matrix, cumulative_target_frequency_matrix, and age_corpus_size_array.
    """
    print('Creating Word Frequences Matrices by Age')
    num_targets = len(target_to_index)
    num_ages = len(age_to_index)

    target_age_freq_matrix = np.zeros([num_targets, num_ages], int)
    cumulative_target_frequency_matrix = np.zeros([num_targets, num_ages], int)
    age_corpus_size_array = np.zeros([num_ages], int)

    for _, _, age, utterance in childes_corpus_data:
        age_index = age_to_index[age]

        for token in utterance:
            token = token.lower()

            if token in target_to_index:
                target_age_freq_matrix[target_to_index[token], age_index] += 1
            age_corpus_size_array[age_index] += 1

    np.cumsum(target_age_freq_matrix, axis=1, out=cumulative_target_frequency_matrix)

    return target_age_freq_matrix, cumulative_target_frequency_matrix, age_corpus_size_array



def co_occurrence_matrix(target_to_index, age_to_index, childes_corpus_data):
    """Create co-occurrence matrices.
    Args:
        target_to_index (dict): A dictionary mapping target words to indices.
        age_to_index (dict): A dictionary mapping ages to indices.
        childes_corpus_data (list): A list containing CHILDES corpus data.

    Returns:
        tuple: A tuple containing cooc_matrix_by_age_list and cumulative_cooc_matrix_by_age_list.
    """
    print('Creating co-occurrence matrices')
    window_type = 'forward'  # forward, backward, summed, concatenated
    window_size = 7
    window_weight = 'flat'  # linear or flat
    PAD = '*PAD*'

    num_targets = len(target_to_index)
    num_ages = len(age_to_index)
    cooc_matrix_by_age_list = []
    cumulative_cooc_matrix_by_age_list = []

    corpus_by_age_list = []
    for i in range(num_ages):
        corpus_by_age_list.append([])

    for i in range(len(childes_corpus_data)):
        utterance = childes_corpus_data[i][3]
        age = childes_corpus_data[i][2]
        age_index = age_to_index[age]
        corpus_by_age_list[age_index] += utterance

    for i in range(num_ages):
        print(f"     Age Index: {i}")
        cooc_matrix = np.zeros([num_targets, num_targets], float)
        cumulative_cooc_matrix = np.zeros([num_targets, num_targets], float)
        current_corpus = corpus_by_age_list[i]
        if len(current_corpus) > 0:
            current_corpus += [PAD] * window_size  # add pad such that all co-occurrences in last window are captured
            windows = itertoolz.sliding_window(window_size, current_corpus)

            for w in windows:
                for word1, word2, dist in zip([w[0]] * (window_size - 1), w[1:], range(1, window_size)):
                    if word1 == PAD or word2 == PAD:
                        continue
                    if word1 not in target_to_index:
                        continue

                    if word2 not in target_to_index:
                        continue

                    word1_index = target_to_index[word1]
                    word2_index = target_to_index[word2]

                    if window_weight == "linear":
                        cooc_matrix[word1_index, word2_index] += window_size - dist
                    elif window_weight == "flat":
                        cooc_matrix[word1_index, word2_index] += 1

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

def calculate_lexical_diversity(age_list, target_list, cumulative_cooc_matrix_by_age_list, target_to_index, age_to_index):
    print('Calculating Lexical Diversity')

    num_targets = len(target_to_index)
    num_ages = len(age_to_index)

    # Initialize an empty matrix to store the lexical diversity values for each target and age
    ld_age_matrix = np.zeros((num_targets, num_ages), dtype=float)

    # Loop through all age groups
    for age_idx in range(num_ages):
        # Loop through all target words
        for target_idx in range(num_targets):
            # Get the co-occurrence values for the current target and age
            current_target_coocs = cumulative_cooc_matrix_by_age_list[age_idx][target_idx, :]

            # Count the number of non-zero co-occurrence values
            num_nonzero = np.count_nonzero(current_target_coocs)

            # Calculate the proportion of non-zero co-occurrence values to the total number of targets
            prop_nonzero = num_nonzero / num_targets

            # Store the calculated proportion in the lexical diversity matrix
            ld_age_matrix[target_idx, age_idx] = prop_nonzero

    return ld_age_matrix


def dd_matrix(target_to_index, age_to_index, doc_to_index, doc_list, childesdb_data):
    num_targets = len(target_to_index)
    num_ages = len(age_to_index)
    num_docs = len(doc_to_index)

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

        age_index = age_to_index[age]

        doc_age_dict[document_id] = age
        if document_id not in encountered_doc_index_dict:
            docs_per_age_matrix[age_index] += 1

        for token in utterance:
            if token in target_to_index:
                if document_id in doc_to_index:
                    target_index = target_to_index[token]
                    document_index = doc_to_index[document_id]
                    target_document_freq_matrix[target_index, document_index] += 1

    cumul_docs_per_age_matrix = np.zeros([num_ages], float)
    cumul_docs_per_age_matrix[0] = docs_per_age_matrix[0]
    for i in range(num_ages - 1):
        cumul_docs_per_age_matrix[i+1] = cumul_docs_per_age_matrix[i] + docs_per_age_matrix[i+1]

    nonzero_sum_matrix = np.zeros([num_targets, num_ages], float)

    for i in range(num_targets):
        for j in range(num_docs):
            if target_document_freq_matrix[i, j] > 0:
                document_id = doc_list[j]
                current_age = doc_age_dict[document_id]
                current_age_index = age_to_index[current_age]

                increment_window_size = num_ages - current_age_index + 1
                for k in range(increment_window_size):
                    try:
                        nonzero_sum_matrix[i, current_age_index + k] += 1
                    except IndexError:
                        print("outofbounds")

    for i in range(num_targets):
        for j in range(num_ages):
            dd_by_age_matrix[i, j] = nonzero_sum_matrix[i, j] / num_docs

    for i in range(num_targets):
        for j in range(num_ages):
            if j == 0:
                dd_by_age_matrix_cumul[i, j] = dd_by_age_matrix[i, j]
            else:
                dd_by_age_matrix_cumul[i, j] = dd_by_age_matrix[i, j] + dd_by_age_matrix_cumul[i, j - 1]

    return dd_by_age_matrix, cumul_docs_per_age_matrix, nonzero_sum_matrix



def get_mcdip_by_age_matrix(age_to_index, target_list, target_to_index, mcdi_data):
    mcdip_df = pd.DataFrame(mcdi_data)

    num_ages = len(age_to_index)
    num_targets = len(target_to_index)

    # Convert mcdip in the dataframe to an age (rows) x mcdi_word (cols) matrix
    mcdip_matrix = np.zeros([num_ages, num_targets], float)
    df_data = mcdi_data.values

    # Create an empty matrix and populate it with corresponding mcdip scores
    for i in range(len(df_data)):
        age = df_data[i][2]
        target = df_data[i][3].lower()
        mcdip = df_data[i][7]
        age_index = age_to_index[age]
        target_index = target_to_index[target]
        mcdip_matrix[age_index, target_index] = mcdip

    return mcdip_matrix


def calculate_kw_cooc(mcdip_matrix, cumulative_cooc_matrix_by_age_list, shuffle=True):
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
    if shuffle: 
        np.random.shuffle(np.transpose(mcdip_values))

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
    f = open('data\\cooc_matrices.txt', 'w')
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
                                                                    cumulative_cooc_matrix_by_age_list)

        correlation_list = compute_mcdip_correlation(prokwo_shuffled_matrix, mcdip_matrix)
        prokwo_shuffled_correlation_matrix[:,i] = np.array(correlation_list)

        if i % 100 == 0:
            print("     Finished {} shuffles".format(i))

    mean_correlations = prokwo_shuffled_correlation_matrix.mean(1)
    
    # Save the shuffled correlations to a csv file
    np.savetxt("data\\prokwo_shuffled_correlations.csv", prokwo_shuffled_correlation_matrix, delimiter=",", fmt = "%10s")
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
                        kwcooc_proportions_transposed,
                        prokwo_shuffle
                       ):
    print('Outputting Target Data')
    #MCDIp = []
    num_targets = len(target_list)
    num_ages = len(age_list)
    f = open('data\\distributional_statistics.csv', 'w')
    f.write('age,word,FQ,LD,DD,PKC\n')
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

    childesdb_data = load_childesdb_data("data\\raw_childes.csv")
    # Load the preprocessed MCDI and target words data.
    mcdi_data, target_words = load_mcdi_data(
        "data\\preprocessed_mcdi.csv",
        "data\\target_words.csv"
    )
    age_list, age_index_dict = create_age_data_structures(mcdi_data)
    target_list, target_index_dict = create_target_data_structures(target_words)

    doc_list, doc_index_dict = create_doc_data_structures(childesdb_data)

    word_list, word_index_dict = create_word_data_structures(childesdb_data)
    target_age_freq_matrix, cumulative_target_frequency_matrix, age_corp_size_array = target_frequency_age_matrix(target_index_dict, age_index_dict, childesdb_data)
    


    cooc_matrix_by_age_list, cumulative_cooc_matrix_by_age_list = co_occurrence_matrix(target_index_dict, age_index_dict, childesdb_data)
 
    for j in range(len(age_list)):
        print("Age {}".format(age_list[j]))
    for i in range(len(target_list)):
        print(target_list[i], cooc_matrix_by_age_list[-1][i,:])

    dd_by_age_matrix, nonzero_sum_matrix, dd_by_age_matrix_cumul = dd_matrix(target_index_dict, age_index_dict, doc_index_dict, doc_list, childesdb_data)
    target_ld_by_age_matrix = calculate_lexical_diversity(age_list, target_list, cumulative_cooc_matrix_by_age_list,target_index_dict,age_index_dict)
    output_cooc_matrices(age_list, target_list, cooc_matrix_by_age_list)


    for i in range(len(target_list)):
        print(target_list[i], dd_by_age_matrix_cumul[i,:])

    mcdip_matrix = get_mcdip_by_age_matrix(age_index_dict, target_list, target_index_dict, mcdi_data)

    prokwo_matrix, prokwo_matrix_transposed=calculate_kw_cooc(mcdip_matrix, 
                                                              cumulative_cooc_matrix_by_age_list)

    correlation_list = compute_mcdip_correlation(prokwo_matrix, mcdip_matrix)
    print(correlation_list)

    shuffle_correlation_list = calculate_prokwo_shuffle_correlations(mcdip_matrix, cumulative_cooc_matrix_by_age_list, 1000)
    print(shuffle_correlation_list)
    prokwo_shuffle = calculate_prokwo_shuffle_correlations(mcdip_matrix, cumulative_cooc_matrix_by_age_list, 1000)



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
                    prokwo_matrix_transposed,
                    prokwo_shuffle
                    )

main()
