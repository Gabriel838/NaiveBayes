from NB import NaiveBayes
from process_data import DataProcess

data = DataProcess(SPACY=True)

# load data
train_text, train_labels = data.load_csv("dataset/smallset/train.csv")
train_tkns = list(map(data.preprocess_text, train_text))
vocab = data.get_vocab(train_tkns)
vocab_size = len(vocab)

test_text, test_labels = data.load_csv("dataset/smallset/test.csv", delimiter=',')
test_tkns = list(map(data.preprocess_text, test_text))
print("test tkns:", test_tkns)

# load model
model = NaiveBayes(vocab)
grouped_tkns = model.categorize_text(train_tkns, train_labels)
print("group tkn:", grouped_tkns)
print("label2idx:", model.label2idx)
print("vocab:", model.vocab)
grouped_counts = model.nb_training(grouped_tkns)
print("counts:", grouped_counts)
preds = model.nb_predicting(test_tkns, grouped_counts)
print("pred:", preds)
print("gold:", test_labels)