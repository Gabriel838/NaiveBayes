from nltk import word_tokenize
import spacy

class DataProcess():
    nlp = None
    SPACY = None

    def __init__(self, SPACY=False):
        self.SPACY = SPACY
        if SPACY == True:
            self.nlp = spacy.load('en_core_web_sm')

    def load_csv(self, file, head=False, delimiter='+'):
        docs, labels = [], []
        with open(file, 'r') as f:
            for i, ln in enumerate(f):
                if head == True and i == 0:
                    continue
                if ln.strip() != "":
                    splited_text = ln.strip().split(delimiter)
                    docs.append(splited_text[0])
                    labels.append(splited_text[1])

        labels = list(map(int, labels))

        return docs, labels


    def preprocess_text(self, text):
        ####################### NLTK method #####################
        if self.SPACY == False:
            # tokenize words
            tokens = word_tokenize(text)

            # lower case
            tokens = list(map(lambda x: x.lower(), tokens))

            # lemma

        ###################### spacy method #####################
        else:
            doc = self.nlp(text)
            token_lemma = [token.lemma_ for token in doc]

        return token_lemma

    def get_vocab(self, tkn_text):
        vocab = set()
        for sent in tkn_text:
            if isinstance(sent, list):
                vocab.update(sent)
            else:
                raise ValueError("Element of tokenized data must be list")

        return vocab


