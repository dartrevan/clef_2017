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

#### Citing:
#####  KFU at CLEF eHealth 2017 Task 1: ICD-10 Coding of English Death Certificates with Recurrent Neural Networks Z Miftakhutdinov, E Tutubalina - 2017
  http://ceur-ws.org/Vol-1866/paper_64.pdf
##### BibTex:
    @inproceedings{
        miftakhutdinov2017kfu,
        title={KFU at CLEF eHealth 2017 Task 1: ICD-10 Coding of English Death Certificates with Recurrent Neural Networks},
        author={Miftakhutdinov, Z and Tutubalina, Elena},
        year={2017},
        organization={CLEF}
    }
