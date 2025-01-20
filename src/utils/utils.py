"""This module contains utility functions and classes for data preprocessing, model training, evauluation, and prediction."""

import copy
import json
import os
import re
import string

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import ollama
import pandas as pd
import tqdm
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from wordcloud import WordCloud

from . import config


class Utils:
    """
    Utility class with helper functions that are used throughout the project
    """
    @staticmethod
    def balance_dataset(df, target):
        """
        Balances the dataset by undersampling the majority classes
        """
        # Get the count of each class in the target column
        target_counts = df[target].value_counts()

        # Get the category with the least amount of samples
        min_count = target_counts.min()

        balanced_dfs = []
        for sentiment, count in target_counts.items():
            df_tmp = df[df[target] == sentiment]
            if count > min_count:
                # Undersample the majority classes
                df_resampled = resample(
                    df_tmp,
                    replace=False,
                    n_samples=min_count,
                    random_state=config.RANDOM_STATE
                )
                balanced_dfs.append(df_resampled)
            else:
                balanced_dfs.append(df_tmp)

        # Concatenate the balanced dataframes
        df_balanced = pd.concat(balanced_dfs)

        return df_balanced

    @staticmethod
    def load_data(drop_neutral=False):
        """
        Load the training and cross-validation datasets
        """
        # Directory containing the prepared datasets
        data_dir = "../res/prepared/"

        # Load the training and cross-validation datasets
        df = pd.read_csv(data_dir + config.TRAIN_SET)
        df_cv = pd.read_csv(data_dir + config.CV_SET)

        # Ensure the target column is of type string
        df[config.TARGET] = df[config.TARGET].astype(str)

        if drop_neutral:
            # Drop the rows with the sentiment neutral
            df = df[df[config.TARGET] != "neutral"]
            df_cv = df_cv[df_cv[config.TARGET] != "neutral"]

        # Update the list of unique sentiments in the config for the confusion matrices
        config.SENTIMENTS = df[config.TARGET].unique().tolist()

        return df, df_cv

    @staticmethod
    def plot_historgram(df, column):
        """
        Plot a histogram of the values in the specified column
        """
        fig, ax = plt.subplots(figsize=(5, 5))

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].plot.hist(ax=ax, bins=50)
        else:
            df[column].value_counts().plot.bar(ax=ax)

        ax.set_title(column.capitalize())
        ax.set_xlabel(column.capitalize())
        ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_wordcloud(df, column, nltk_extra_stopwords=False):
        """
        Plot a wordcloud of the words in the specified column
        """
        # Tokenize the text in the specified column
        tokenizer = NLTKTokenizer(nltk_extra_stopwords)
        words = []

        # Tokenize the text in the specified column
        df.apply(lambda row: words.extend(tokenizer.tokenize(row[column]).split(" ")), axis=1)

        # Count the frequency of each word
        word_freq = pd.Series(words).value_counts()
        print(word_freq)

        # Create the wordcloud
        wordcloud = WordCloud(width=2000, height=1000, background_color='white')
        wordcloud.generate_from_frequencies(word_freq)

        # Plot wordcloud
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud)
        plt.axis('off')

        return word_freq


class NLTKTokenizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that cleanses, tokenizes and lemmatizes text using the NLTK library
    """
    def __init__(self, extra_stop_words=False, lemmatize=False, remove_urls=False):
        self.extra_stop_words = extra_stop_words
        self.lemmatize = lemmatize
        self.remove_urls = remove_urls

        # Download the required NLTK resources
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download('wordnet', quiet=True)

        # Load the stop words
        self.stop_words = set(nltk.corpus.stopwords.words("english"))
        self.crypto_words = set()
        self.company_words = set()

        # Load the custom stop words
        if self.extra_stop_words:
            with open("../res/stopwords.txt", "r") as f:
                self.stop_words.update(f.read().splitlines())

            with open("../res/crypto_words.txt", "r") as f:
                self.crypto_words.update(f.read().splitlines())

            with open("../res/company_words.txt", "r") as f:
                self.company_words.update(f.read().splitlines())

        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def fit(self, x, y=None):
        """
        Fit method to return the transformer object
        """
        return self

    def transform(self, x):
        """
        Transform method to apply the tokenization to the text
        """
        return x.apply(self.tokenize)

    def replace_substrings(self, text, words_to_replace, replace_with):
        """
        Replace substrings in a text with a specified string
        """
        sentence = text.split(" ")
        for i in range(len(sentence)):
            if sentence[i] in words_to_replace:
                sentence[i] = replace_with

        return " ".join(sentence)

    def tokenize(self, text):
        # Normalization
        text = str(text)

        # Encoding errors
        encoding_errors = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "â€˜": "'",
            "â€”": "-",
            "â€“": "-",
            "â€¢": "•",
            "â€¦": "...",
            "\\xa0": "",
            "\\x9d": "",
            "\\x9c": "",
            "\\x9f": "",
            "\\x8f": "",
            "\\x99": "",
            "\\x93": "",
            "\\x92": "",
            "\\x91": "",
            "\\x96": "",
            "\\x94": "",
            "\\x85": "",
            "\\x82": "",
            "\\x81": "",
            "\\x80": "",
            "\\x9a": "",
            "\\x87": "",
            "\\x86": "",
            "\\x84": "",
            "\\x83": "",
            r"ÃƒÂ¼": "ü",
            r"\r": "",
            r"\n": "",
            r"\t": "",
        }

        # Replace encoding errors
        for key, value in encoding_errors.items():
            text = text.replace(key, value)

        # Replace custom stop words with a placeholder
        if self.extra_stop_words:
            text = self.replace_substrings(text, self.crypto_words, 'TICKER')
            text = self.replace_substrings(text, self.company_words, 'TICKER')

        # Remove URLs from text
        if self.remove_urls:
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            text = url_pattern.sub("", text)

        # Ensure text only consists of alphanumeric characters
        text = re.sub(r'\W', ' ', str(text))

        # Tokenization
        tokens = nltk.word_tokenize(text.lower())

        # Stopwords
        filtered_tokens = [
            t for t in tokens if t not in self.stop_words and t not in string.punctuation]

        # Lemmatization
        if self.lemmatize:
            # POS-Tagging and Lemmatization
            filtered_tokens = [
                self.lemmatize_with_pos(token, pos)
                for token, pos in nltk.pos_tag(filtered_tokens)
                # if not pos.startswith('N')
            ]

        return ' '.join(filtered_tokens)

    def lemmatize_with_pos(self, token, pos):
        """
        Lemmatize the token with the specified POS-tag
        """
        pos = self.get_wordnet_pos(pos)
        if pos:
            return self.lemmatizer.lemmatize(token, pos=pos)
        else:
            return self.lemmatizer.lemmatize(token)

    def get_wordnet_pos(self, nltk_pos):
        """
        Map the NLTK POS-tags to the WordNet POS-tags
        """
        if nltk_pos.startswith('J'):
            return 'a'  # Adjectives
        elif nltk_pos.startswith('V'):
            return 'v'  # Verbs
        elif nltk_pos.startswith('N'):
            return 'n'  # Nouns
        elif nltk_pos.startswith('R'):
            return 'r'  # Adverbs
        else:
            return None


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that transforms text into word vectors using a Word2Vec model
    """
    def __init__(self, model_name='word2vec-google-news-300'):
        self.model_name = model_name
        self.model = KeyedVectors.load_word2vec_format(config.MODEL_DIR + model_name, binary=True)

    def fit(self, X, y=None):
        """
        Fit method to return the transformer object
        """
        return self

    def transform(self, X):
        """
        Transform method to convert text into word vectors
        """
        return np.array([self._average_vector(text) for text in X])

    def _average_vector(self, text):
        """
        Calculate the average word vector for a given text
        """
        words = text.split()
        vectors = [self.model[word] for word in words if word in self.model]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)


class CompoundWordSplitter(BaseEstimator, TransformerMixin):
    """
    Custom transformer that splits compound words into their constituent parts
    """
    def __init__(self):
        pass

    def __split_compound(self, word):
        # Simple Heuristic for segmenting compound words
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", word)

    def fit(self, X, y=None):
        """
        Fit method to return the transformer object
        """
        return self

    def split_compounds(self, text):
        """
        Split compound words in a given text
        """
        words = text.split()
        split_words = []
        for word in words:
            split_words.extend(self.__split_compound(word))
        return " ".join(split_words)

    def transform(self, x):
        """
        Transform method to split compound words in the text
        """
        return x.apply(self.split_compounds)


class CustomPipeline:
    """
    Custom pipeline class that trains a model, evaluates its performance, and dumps it via joblib
    """

    def __init__(self, df, features, target, steps, df_cv=None, model_name=None):
        self.df = df  # Training data
        self.df_cv = df_cv  # Cross-validation data
        self.features = features
        self.target = target

        self._model_name = model_name

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

        self._X_train_balanced = None
        self._X_test_balanced = None
        self._y_train_balanced = None
        self._y_test_balanced = None

        self.pipeline = Pipeline(steps=steps, verbose=True)  # Pipeline for model with unbalanced train data
        self.pipeline_balanced = copy.deepcopy(self.pipeline)  # Pipeline for model with balanced train data

        self.train_test_split()

    def train_test_split(self):
        """
        Split the dataset into training and testing sets with and without balancing
        """
        # Balance the dataset
        df = Utils.balance_dataset(self.df, self.target)

        # Balanced train-test split
        (
            self._X_train_balanced,
            self._X_test_balanced,
            self._y_train_balanced,
            self._y_test_balanced,
        ) = train_test_split(
            df[self.features],
            df[self.target],
            test_size=0.2,
            random_state=config.RANDOM_STATE,
        )

        # Unbalanced train-test split
        df = self.df
        self._X_train, self._X_test, self._y_train, self._y_test = (
            train_test_split(
                df[self.features],
                df[self.target],
                test_size=0.2,
                random_state=config.RANDOM_STATE,
            )
        )

    def fit(self, balance):
        """
        Train the model with balanced or unbalanced data
        """
        if balance:
            self.pipeline_balanced.fit(
                self._X_train_balanced, self._y_train_balanced)
        else:
            self.pipeline.fit(self._X_train, self._y_train)

    def evaluate(self, balanced_model):
        """
        Evaluate model performance and display confusion matrices
        """
        if balanced_model:
            model = self.pipeline_balanced
        else:
            model = self.pipeline

        X_test = self._X_test
        y_test = self._y_test
        y_pred = model.predict(X_test)

        model_name = f'{self._model_name} - {"balanced" if balanced_model else "unbalanced"} train data'
        print(f"Classification Report for {model_name}:")

        # Plotting confusion matrices
        if self.df_cv is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Confusion Matrix for Test Data
            cm_test = confusion_matrix(y_test, y_pred, normalize="true", labels=config.SENTIMENTS)
            disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=config.SENTIMENTS)
            disp_test.plot(ax=axes[0], cmap=plt.cm.Blues)
            axes[0].set_title(f"Test Data - {model_name}")

            # Confusion Matrix for Cross-Validation Data
            X_cv = self.df_cv[self.features]
            y_cv = self.df_cv[self.target]
            y_cv_pred = model.predict(X_cv)

            cm_cv = confusion_matrix(
                y_cv, y_cv_pred, normalize="true", labels=config.SENTIMENTS)
            disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=config.SENTIMENTS)
            disp_cv.plot(ax=axes[1], cmap=plt.cm.Blues)
            axes[1].set_title(f"CV Data - {model_name}")

            print("Classification Report - Validation set\n", classification_report(y_test, y_pred))
            print("Classification Report - Crossvalidation set\n", classification_report(y_cv, y_cv_pred))

            plt.tight_layout()
            plt.show()

        else:
            # Only Test Data Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, normalize="true", labels=config.SENTIMENTS)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.SENTIMENTS)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Normalized Confusion Matrix for {model_name}")
            plt.show()

            print(classification_report(y_test, y_pred))

    def predict(self, X, balanced_model):
        """
        Make predictions using the trained model (balanced or unbalanced)
        """
        if balanced_model:
            model = self.pipeline_balanced
        else:
            model = self.pipeline

        return model.predict(X)

    def dump(self, path=None, name=None):
        """
        Dump the model to disk using joblib
        """
        if name is None:
            name = self._model_name
        if path is None:
            path = config.MODEL_DIR
        
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self, f"{path}/{name}.joblib")


class OllamaPipeline(CustomPipeline):
    """
    Custom pipeline that uses the Ollama API to predict sentiment from text
    """
    def __init__(self, df, features, target, ollama_model, model_name=None):
        steps = []
        super().__init__(df, features, target, steps, model_name)

        # Download the specified Ollama model
        self.ollama_model = ollama_model
        ollama.pull(self.ollama_model)
        # System instructions for the Ollama prompt
        self.system_instructions = (
            "You are a helpful assistant specialized in analyzing news articles. "
            "Provide concise and accurate responses in JSON format."
        )

    def fit(self):
        """
        Train the model using the Ollama API
        """
        raise NotImplementedError("OllamaPipeline: No training required for this pipeline.")

    def evaluate(self):
        """
        Evaluate the models performance using the test set
        """
        model_name = self._model_name

        X_test = self._X_test
        y_test = self._y_test
        y_pred = self.predict(X_test)
        y_pred = pd.Series(y_pred, index=y_test.index)

        # Remove invalid predictions
        y_test_bak = y_test.copy(deep=True)
        valid_indices = y_pred.apply(lambda x: x in config.SENTIMENTS)
        y_test = y_test[valid_indices].reset_index(drop=True)
        y_pred = y_pred[valid_indices].reset_index(drop=True)
        print(f"There were {len(y_test_bak) - len(y_test)}/{len(y_test_bak)} invlaid predictions. Removed them from the evaluation. New test set size: {len(y_test)}")

        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred, labels=config.SENTIMENTS))

        print(f"Confusion Matrix for {model_name}:")
        cm = confusion_matrix(y_test, y_pred, labels=config.SENTIMENTS, normalize="true")
        print(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.SENTIMENTS)
        disp.plot(cmap=plt.cm.Blues)

        plt.title(f"Normalized Confusion Matrix for {model_name}")
        plt.show()


    def predict(self, X):
        """
        Make predictions using the Ollama API
        """
        errors = []
        y_pred = []

        for index, row in tqdm.tqdm(X.iterrows(), desc="Analyzing sentiment with Ollama"):
            try:
                # Generate the prompt
                prompt = (
                    f"{self.system_instructions}\n\n"
                    f"Analyze the following blog article and return a JSON object with the following fields:\n"
                    f"main_core_thoughts (a summary of the main core thoughts),\n"
                    f"sentiment_analysis (sentiment as positive, neutral, or negative),\n"
                    f"confidence_level (confidence score from 0 to 1),\n"
                    f"topics (a list of up to 3 main topics discussed in the blog article).\n\n"
                )
                for feature in self.features:
                    prompt += f"{feature}: {row[feature]}\n"

                # Generate the response
                response = ollama.generate(model=self.ollama_model, prompt=prompt, stream=False, format="json")
                response = json.loads(response.model_dump_json())
                response = json.loads(response["response"])["sentiment_analysis"]

                y_pred.append(response)

            except Exception as e:
                y_pred.append("error")
                errors.append((index, str(e)))

        for error in errors:
            print(f"Error for index {error[0]}: {error[1]}")

        return y_pred
