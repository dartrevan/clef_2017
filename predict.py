from argparse import ArgumentParser
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pandas import read_csv, DataFrame
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import pickle
from distances import cosine_similarities
import codecs


np.random.seed(42)
token_index = {}
icd_codes = []
input_max_len = 34
dictionary_icd_codes_count = 4350


def read_data(data_path):
    df = read_csv(data_path, sep=';')
    return df.RawText.values, df[['DocID', 'YearCoded', 'LineID']]



def map_one_hot(one_hot): return icd_codes[np.argmax(one_hot)]
def map_icd_code(icd_code):
    one_hot = np.zeros(len(icd_codes))
    one_hot[icd_codes.index(icd_code)] = 1
    return one_hot



def toDF(index, sequences):
    df = []
    for row, seq in zip(index.iterrows(), sequences):
        row_index, ind = row
        for i, code in enumerate(seq):
            if code == 'EOS': break
            df.append({
                'DocID':ind['DocID'],
                'YearCoded':ind['YearCoded'],
                'LineID':ind['LineID'],
                'Rank':i + 1,
                'StandardText': None,
                'ICD10':code,
                })

    return DataFrame(df)





if __name__ == '__main__':
    parser = ArgumentParser(description='This script produces ICD-10 codes for death certificates (provided by CLEF eHealth task 1) using neural networks')
    parser.add_argument('--nn_model', dest='nn_model', type=unicode, help='path to neural network model', default='nn_models/seq2seq.bin')
    parser.add_argument('--token_index', dest='token_index', type=unicode, help='path to token index', default='nn_models/seq2seq_token_index.bin')
    parser.add_argument('--icd_codes', dest='icd_codes', type=unicode, help='path icd codes mappings', default='nn_models/seq2seq_icd_codes.bin')

    parser.add_argument('-c', '--certificates', dest='certificates_path', type=unicode, help='path to csv file with death certificates')

    args = parser.parse_args()
    nn_model_path = args.nn_model
    token_index_path = args.token_index
    icd_codes_path = args.icd_codes
    certificates_path = args.certificates_path


    #loading trained model
    seq2seq_net = load_model(nn_model_path)
    with codecs.open(token_index_path, 'rb') as in_file:
        token_index = pickle.load(in_file)
    with codecs.open(icd_codes_path, 'rb') as in_file:
        icd_codes = pickle.load(in_file)


    # preparing input data
    raw_texts, index = read_data(certificates_path)
    similarities = np.array([sim for sim in cosine_similarities(raw_texts)])
    input_sequences = [wordpunct_tokenize(raw_text.lower()) for raw_text in raw_texts]
    input_sequences = np.array([[token_index[token] if token in token_index else token_index['UNKNOWN'] for token in document] for document in input_sequences])
    input_sequences = pad_sequences(input_sequences, input_max_len, padding='post', truncating='post', value=token_index['EOS'])


    """
        predicting
    """
    output_sequences_predicted = seq2seq_net.predict({'token_indexes':input_sequences, 'similarities_in':similarities})
    output_sequences_predicted = [[map_one_hot(one_hot) for one_hot in document] for document in output_sequences_predicted]
    toDF(index, output_sequences_predicted)[['DocID', 'YearCoded', 'LineID', 'Rank', 'StandardText', 'ICD10']].to_csv('run1.csv', index=False, sep=';')
