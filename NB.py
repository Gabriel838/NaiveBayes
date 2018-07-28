"""
Take away: 1. How to handle unseen words
              if words not appear in training data:
                    if words not appear in class:
                        map to UNK (to solve UNK's 0 probability problem, use smooth)
              else:
                    ignore

           2. Log probability
              log probability is used for calculating the most likable class of an individual instance

Probability cal:
    P(C|X) = P(X|C) P(C) / sum_c P(X, c)
           = P(doc|C) P(C) / sum_c P(X, c)
           = P(w_1, w_2, ..., w_n|C) P(C) / sum_c P(X, c)
           = P(w_1|C) P(w_2 | w_1, C) ... P(w_n | w_1, w_2, ..., w_n, C) P(C) / sum_c P(X, c)
           = P(w_1|C) P(w_2 | C) ... P(w_n | C) P(C) / sum_c P(X, c)
    use log, then we have
    log P(C|X) = sum_i log P(w_i | C) + log P(C) - log(sum_c P(X,c))
"""

from collections import Counter
import math

from process_data import DataProcess




class NaiveBayes(object):
    # not using unknown word model
    vocab = None
    num_labels = None
    idx2label = None
    label2idx = None
    priors = None
    doc_size = None

    def __init__(self, vocab):
        self.vocab = vocab

    def categorize_text(self, text_tokens, labels):
        # set attributes
        label_set = set(labels)
        self.num_labels = len(label_set)
        self.idx2label = {i: v for i, v in enumerate(label_set)}
        self.label2idx = {v: k for k, v in self.idx2label.items()}

        # tokens grouped by class
        groups = [[] for _ in range(self.num_labels)]
        for i in range(len(labels)):
            idx = self.label2idx[labels[i]]
            groups[idx] += text_tokens[i]

        # get priors
        priors = [[] for _ in range(self.num_labels)]
        label_counts = Counter(labels)
        for k, v in label_counts.items():
            idx = self.label2idx[k]
            priors[idx] = v / len(labels)
        self.priors = priors

        # get doc size
        doc_size = [len(g) for g in groups]
        self.doc_size = doc_size

        return groups

    def nb_training(self, grouped_tkns):
        grouped_probs = [Counter(doc) for doc in grouped_tkns]
        for i, c in enumerate(grouped_probs):
            for k, v in c.items():
                # update probability
                c[k] = (v + 1) / (self.doc_size[i] + len(self.vocab))
            c['UNK'] = 1 / (self.doc_size[i] + len(self.vocab))
            grouped_probs[i] = c

        return grouped_probs


    def nb_predicting(self, tokenized_test, group_counts):
        preds = []
        for doc in tokenized_test:
            doc_probs = []
            for i, class_counts in enumerate(group_counts):
                c_likelihood = self.log_prob(doc, class_counts)
                c_prob = math.log(self.priors[i]) + c_likelihood
                doc_probs.append(c_prob)
            print("inter pred:", doc_probs)

            idx, _ = max(list(enumerate(doc_probs)), key=lambda x: x[1])
            preds.append(self.idx2label[idx])
        return preds


    def log_prob(self, doc, class_counts):
        log_prob = 1
        for word in doc:
            # ignore words not in train
            if word in self.vocab:
                if word in class_counts:
                    print("word:", word, "prob:", class_counts[word])
                    log_prob += math.log(class_counts[word])
                    # log_prob *= counts[word]
                else:
                    log_prob += math.log(class_counts['UNK'])
                    # log_prob *= counts['UNK']
            else:
                pass

        return log_prob



if __name__ == "__main__":
    # data
    train_data = ["just plain boring",
                  "entirely predictable and lacks energy",
                  "no surprises and very few laughs",
                  "very powerful",
                  "the most fun film of the summer"]
    train_labels = [-1, -1, -1, 1, 1]
    test_data = ["predictable with no fun"]

    # load data
    data = DataProcess(SPACY=True)
    tkn_train = list(map(data.preprocess_text, train_data))
    vocab = data.get_vocab(tkn_train)
    vocab_size = len(vocab)
    tkn_test = list(map(data.preprocess_text, test_data))

    model = NaiveBayes(vocab)
    grouped_tkns = model.categorize_text(tkn_train, train_labels)
    grouped_counts = model.nb_training(grouped_tkns)
    preds = model.nb_predicting(tkn_test, grouped_counts)
    print("pred:", preds)

