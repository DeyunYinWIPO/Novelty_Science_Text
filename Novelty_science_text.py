# Python code for 
# Step1: Training Word Embeddings with all text data in Web of Science
# Step2: Computing indicators 


import pandas as pd
import numpy as np
import time
from multiprocessing import Process
from gensim.models import KeyedVectors, word2vec, Word2Vec


# Train a Word2Vec embedding model
def train_word2vec():
    # Read sentences from the 'scientific_pubs.txt' file, where each line is a sentence
    sentences = word2vec.LineSentence('scientific_pubs.txt')

    # Initialize and train the Word2Vec model
    # vector_size: Dimensionality of the word vectors
    # min_count: Ignores words with total frequency lower than this
    # window: Maximum distance between the current and predicted word within a sentence
    # sg: Training algorithm; 0 for CBOW, 1 for Skip-gram
    model = Word2Vec(sentences, vector_size=300, min_count=5, window=5, sg=0)

    # Save the trained model to the specified path
    model.save('../trained_word2vec.model')


# Convert word vectors to a document (sentence) vector
def get_sentence_matrix(model, splited_words):
    # Load the pre-trained Word2Vec model
    model = Word2Vec.load('../trained_word2vec.model')

    # Initialize an empty DataFrame to store word vectors
    words_matrix = pd.DataFrame({})

    # Iterate over each word in the split words list
    for i in range(len(splited_words)):
        word = splited_words[i]
        # Check if the word exists in the model's vocabulary
        if word in model.wv.index_to_key:
            # Add the word vector to the DataFrame
            words_matrix[word] = np.array(model.wv[word])

    # Compute the mean vector across all word vectors to get the sentence vector
    words_vector = words_matrix.astype('float').mean(axis=1)

    # Return the sentence vector as a list
    return words_vector.tolist()


# Calculate element novelty based on vector similarity
def get_element(vec_focal_calcu, vec_backward):
    # Compute the dot product (similarity) between focal vectors and backward vectors
    data_sim = np.dot(vec_focal_calcu, vec_backward)

    # Convert similarity to dissimilarity
    data_sim = 1 - data_sim

    # Initialize an empty DataFrame to store the results
    data_result = pd.DataFrame({})

    # Calculate the minimum dissimilarity for each focal vector
    element_0 = np.min(data_sim, axis=1)

    # Calculate the average dissimilarity for each focal vector
    average = np.mean(data_sim, axis=1)

    # Calculate specific percentiles of dissimilarity for each focal vector
    element_1, element_5, element_10, element_25, element_50 = np.percentile(
        data_sim, q=[1, 5, 10, 25, 50], axis=1)

    # Assign the calculated metrics to the DataFrame
    data_result['element_0'] = element_0
    data_result['element_1'] = element_1
    data_result['element_5'] = element_5
    data_result['element_10'] = element_10
    data_result['element_25'] = element_25
    data_result['element_50'] = element_50

    # Return the result DataFrame
    return data_result


# Function to perform element novelty calculation using multiprocessing
def get_element_multiprocessing(vec_start, vec_end, filename, chunk_size):
    # Load the backward vectors from a NumPy file and transpose for compatibility
    vec_backward = np.load('../data/vec_backward/vec1998_2017.npy').T

    # Print the start of the processing with relevant information
    print(r'%s start:focal length=%s backward length=%s %s:%s' % (
        filename,
        vec_end - vec_start,
        vec_backward.shape[1],
        vec_start,
        vec_end
    ))

    # Iterate over chunks of the focal vectors
    for i in range(int((vec_end - vec_start) / chunk_size)):
        # Record the start time for performance measurement
        time_start = time.time()

        # Determine the start and end indices for the current chunk
        vec_chunk_start = vec_start + i * chunk_size
        vec_chunk_end = vec_start + (i + 1) * chunk_size

        # Load the focal vectors for the current chunk from a NumPy file
        vec_focal = np.load('../data/vec_focal/%s.npy' % filename)[vec_chunk_start:vec_chunk_end]

        # Calculate the novelty metrics for the current chunk
        percentile_result = get_element(vec_focal, vec_backward)

        # Save the results to a CSV file with a naming convention that includes the chunk indices
        percentile_result.to_csv(
            '../data/novelty/result_vector20230915/novelty_%s_%s_%s.csv' % (
                filename, vec_chunk_start, vec_chunk_end),
            index=False
        )

        # Record the end time and calculate the time taken for processing the chunk
        time_end = time.time()
        time_cost = time_end - time_start

        # Print the completion message with time cost
        print('%s %s:%s time cost:%s ' % (
            filename, vec_chunk_start, vec_chunk_end, time_cost))


# Main execution block
if __name__ == '__main__':
    # Define the filename for focal vectors
    filename = 'vector'

    # Load the focal vectors from a NumPy file
    focal = np.load('../data/vec_focal/%s.npy' % filename)

    # Define the number of processes for multiprocessing
    process_num = 10

    # Define the size of each chunk to be processed by a single process
    chunk_size = 1000

    # Calculate the interval of vectors each process will handle
    interval = int((int(focal.shape[0] / chunk_size) + 1) / process_num) + 1

    # Create and start multiple processes for parallel processing
    for num in range(process_num):
        # Initialize a new process targeting the get_element_multiprocessing function
        p = Process(
            target=get_element_multiprocessing,
            args=(
                num * interval * chunk_size,  # Start index
                (num + 1) * interval * chunk_size,  # End index
                filename,  # Filename
                chunk_size  # Chunk size
            )
        )
        p.start()  # Start the process