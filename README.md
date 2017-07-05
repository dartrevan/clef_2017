### This code uses pretrained recurrent neural network models for normalizing free-text descriptions of causes of death into icd-10 codes. Code written in python 2.7

### Installing required libraries:
    pip install -r requirements.txt

After installing libraries nltk punkt model must be downloaded
    in python interpreter:
        import nltk
        nltk.download('punkt')

Unpack dictionary_vectors.zip

### Usage:
    python predict.py -c file_path
    file_path is a path to the file with death certificates in CLEF eHealth Task 1 format


#### Note: For additional scripts used for training RNNs please contact zulfatmi@gmail.com
