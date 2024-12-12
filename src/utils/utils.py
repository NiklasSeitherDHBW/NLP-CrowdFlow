"""_summary_
"""

import copy
import re
import string
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import csr_matrix  
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import spacy
import nltk
from . import config
from keras.preprocessing.text import Tokenizer
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, FastText



class Utils:
    """_summary_
    """
    @staticmethod
    def balance_dataset(df, target):
        """_summary_

        Args:
            df (_type_): _description_
            features (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Create a second training dataset with balanced classes
        target_counts = df[target].value_counts()

        # Get the category with the least amount of samples
        max_count = target_counts.max()

        balanced_dfs = []
        for sentiment, count in target_counts.items():
            df_tmp = df[df[target] == sentiment]
            if count < max_count:
                df_resampled = resample(
                    df_tmp, replace=True, n_samples=max_count, random_state=config.RANDOM_STATE
                )
                balanced_dfs.append(df_resampled)
            else:
                balanced_dfs.append(df_tmp)

        df_balanced = pd.concat(balanced_dfs)

        return df_balanced

class NLTKTokenizer(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    def __init__(self):
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def fit(self, x, y=None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self

    def transform(self, x):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        return x.apply(self.tokenize)

    def tokenize(self, text):
        """_summary_

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """
        text = re.sub(r'\W', ' ', str(text))
        tokens = nltk.tokenize.word_tokenize(text.lower())
        filtered_tokens = [t for t in tokens if t not in self.stop_words and t not in string.punctuation]
        return ' '.join(filtered_tokens)

import re
from collections import Counter, defaultdict
import math

class BytePairEncoding:
    def __init__(self, vocab, num_merges):
        self.vocab = vocab
        self.num_merges = num_merges

    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair):
        new_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word in self.vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = self.vocab[word]
        return new_vocab

    def encode(self):
        for _ in range(self.num_merges):
            pairs = self.get_stats()
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best)
        return self.vocab

class WordPiece:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {"[UNK]": 0}

    def build_vocab(self, data):
        token_counts = Counter(data)
        while len(self.vocab) < self.vocab_size:
            new_token = max(token_counts, key=token_counts.get)
            self.vocab[new_token] = len(self.vocab)
            del token_counts[new_token]
        return self.vocab

    def tokenize(self, text):
        tokens = []
        for word in text.split():
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append("[UNK]")
        return tokens

class UnigramLanguageModel:
    def __init__(self):
        self.model = {}

    def train(self, corpus):
        total_words = sum(len(sentence.split()) for sentence in corpus)
        word_counts = Counter(word for sentence in corpus for word in sentence.split())
        self.model = {word: count / total_words for word, count in word_counts.items()}

    def tokenize(self, text):
        tokens = []
        for word in text.split():
            if word in self.model:
                tokens.append(word)
            else:
                tokens.append("[UNK]")
        return tokens

class SpacyTokenizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text):
        return [token.text for token in self.nlp(text)]

class FinBERTTokenizer:
    # FinBERT Tokenizer (using Hugging Face Transformers)
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    

class GensimTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        return simple_preprocess(text)

class KerasTokenizer:
    def __init__(self, num_words=None):
        self.tokenizer = Tokenizer(num_words=num_words)

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def tokenize(self, text):
        return self.tokenizer.texts_to_sequences([text])[0]

if __name__ == "__main__":
    # Byte Pair Encoding
    vocab = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}
    bpe = BytePairEncoding(vocab, num_merges=10)
    print("BPE Result:", bpe.encode())

    # WordPiece
    wordpiece = WordPiece(vocab_size=10)
    wordpiece_vocab = wordpiece.build_vocab(["new", "news", "newest", "wide", "widen"])
    print("WordPiece Vocab:", wordpiece_vocab)
    print("WordPiece Tokens:", wordpiece.tokenize("new wide unknown"))

    # Unigram Language Model
    corpus = ["this is a test", "this test is a test"]
    unigram = UnigramLanguageModel()
    unigram.train(corpus)
    print("Unigram Model:", unigram.model)
    print("Unigram Tokens:", unigram.tokenize("this is unknown"))

    # SpaCy
    spacy_tokenizer = SpacyTokenizer()
    print("SpaCy Tokens:", spacy_tokenizer.tokenize("This is a test sentence."))

    # FinBERT
    finbert_tokenizer = FinBERTTokenizer()
    print("FinBERT Tokens:", finbert_tokenizer.tokenize("Stocks are performing well."))

    gensim_tokenizer = GensimTokenizer()
    print("Gensim Tokens:", gensim_tokenizer.tokenize("This is a test sentence."))

    # Keras
    keras_tokenizer = KerasTokenizer(num_words=10)
    texts = ["This is a test", "Another test sentence"]
    keras_tokenizer.fit(texts)
    print("Keras Tokens:", keras_tokenizer.tokenize("This is another test"))

class SparseToDenseBatchTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size):
        self.batch_size = batch_size  # Größe der Batches
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, csr_matrix):
            raise ValueError("Input must be a sparse CSR matrix.")
        
        n_samples = X.shape[0]
        dense_batches = []
        
        # Sparse-Matrix in Batches aufteilen und konvertieren
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            dense_batches.append(X[start:end].toarray())  # Konvertiert nur einen Teil
        
        # Batches wieder zu einer Gesamtmatrix zusammenfügen
        return np.vstack(dense_batches)
    
class CompoundWordSplitter(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    def __init__(self):
        pass

    def __split_compound(self, word):
        """_summary_

        Args:
            word (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Simple Heuristic for segmenting compound words
        # This is a placeholder and should be replaced with a more robust method
        return re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', word)

    def fit(self, X, y=None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self

    def split_compounds(self, text):
        """_summary_

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """
        words = text.split()
        split_words = []
        for word in words:
            split_words.extend(self.__split_compound(word))
        return ' '.join(split_words)

    def transform(self, x):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        return x.apply(self.split_compounds)

class BagOfWords:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def fit_transform(self, corpus):
        return self.vectorizer.fit_transform(corpus).toarray()

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

class Word2VecModel:
    def __init__(self, size=100, window=5, min_count=1, workers=4):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def train(self, sentences):
        self.model = Word2Vec(sentences, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers)

    def get_vector(self, word):
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        else:
            return None

class CBOWModel:
    def __init__(self, size=100, window=5, min_count=1, workers=4):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def train(self, sentences):
        self.model = Word2Vec(sentences, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers, sg=0)

    def get_vector(self, word):
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        else:
            return None

class GloVeModel:
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.word_to_vec = {}

    def train(self, corpus):
        vocab = set(word for sentence in corpus for word in sentence)
        co_occurrence = {word: {w: 0 for w in vocab} for word in vocab}
        for sentence in corpus:
            for i, word in enumerate(sentence):
                for j in range(max(i - 5, 0), min(i + 5 + 1, len(sentence))):
                    if i != j:
                        co_occurrence[word][sentence[j]] += 1
        # Create GloVe vectors
        word_embeddings = {word: np.random.rand(self.embedding_dim) for word in vocab}
        for word in vocab:
            word_embeddings[word] = np.mean([np.array(word_embeddings[w]) for w in co_occurrence[word]], axis=0)
        self.word_to_vec = word_embeddings

    def get_vector(self, word):
        return self.word_to_vec.get(word, None)

class FastTextModel:
    def __init__(self, size=100, window=5, min_count=1, workers=4):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def train(self, sentences):
        self.model = FastText(sentences, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers)

    def get_vector(self, word):
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        else:
            return None

class CustomPipeline:
    """_summary_
    """
    def __init__(self, df, features, target, steps,  model_name=None):
        self.df = df
        self.features = features
        self.target = target
        self.__model_name = model_name

        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None

        self.__X_train_balanced = None
        self.__X_test_balanced = None
        self.__y_train_balanced = None
        self.__y_test_balanced = None

        self.pipeline = Pipeline(
            steps=steps,
            verbose=True
        )

        self.pipeline_balanced = copy.deepcopy(self.pipeline)

    def fit(self, balance):
        """_summary_
        """
        if balance:
            df = Utils.balance_dataset(self.df, self.target)
            
            self.__X_train_balanced, self.__X_test_balanced, self.__y_train_balanced, self.__y_test_balanced = \
                train_test_split(df[self.features], df[self.target], test_size=0.2, random_state=config.RANDOM_STATE)
            self.pipeline_balanced.fit(self.__X_train_balanced, self.__y_train_balanced)

        else:
            
            df = self.df
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = \
                train_test_split(df[self.features], df[self.target], test_size=0.2, random_state=config.RANDOM_STATE)
            self.pipeline.fit(self.__X_train, self.__y_train)

    def evaluate(self, balanced_model, balanced_test_data):
        """_summary_
        """
        if balanced_model:
            model = self.pipeline_balanced
        else:
            model = self.pipeline
            
        if balanced_test_data:
            X_test = self.__X_test_balanced
            y_test = self.__y_test_balanced
        else:
            X_test = self.__X_test
            y_test = self.__y_test

        y_pred = model.predict(X_test)

        model_name = f'{self.__model_name}_{"balanced" if balanced_model else "unbalanced"} model_{"balanced" if balanced_test_data else "unbalanced"} test data'

        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))

        print(f"Confusion Matrix for {model_name}:")
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        print(cm)

        # Display normalized confusion matrix as a graphic
        labels = sorted(list( set(y_test))) #set(y_train) |
        print(labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)

        plt.title(f"Normalized Confusion Matrix for {model_name}")
        plt.show()
