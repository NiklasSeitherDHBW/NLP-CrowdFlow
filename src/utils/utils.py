"""_summary_"""

import copy
import json
import re
import string

import joblib
import matplotlib.pyplot as plt
import nltk
import ollama
import pandas as pd
import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from . import config


class Utils:
    """_summary_"""

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
                    df_tmp,
                    replace=True,
                    n_samples=max_count,
                    random_state=config.RANDOM_STATE,
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
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

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
        text = re.sub(r"\W", " ", str(text))
        tokens = nltk.tokenize.word_tokenize(text.lower())
        filtered_tokens = [
            t
            for t in tokens
            if t not in self.stop_words and t not in string.punctuation
        ]
        return " ".join(filtered_tokens)


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
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", word)

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
        return " ".join(split_words)

    def transform(self, x):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        return x.apply(self.split_compounds)


class CustomPipeline:
    """_summary_"""

    def __init__(self, df, features, target, steps, model_name=None):
        self.df = df
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

        self.pipeline = Pipeline(steps=steps, verbose=True)
        self.pipeline_balanced = copy.deepcopy(self.pipeline)

        self.train_test_split()

    def train_test_split(self):
        df = Utils.balance_dataset(self.df, self.target)

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
        """_summary_"""
        if balance:
            self.pipeline_balanced.fit(self._X_train_balanced, self._y_train_balanced)
        else:
            self.pipeline.fit(self._X_train, self._y_train)

    def evaluate(self, balanced_model, balanced_test_data):
        """_summary_"""
        if balanced_model:
            model = self.pipeline_balanced
        else:
            model = self.pipeline

        if balanced_test_data:
            X_test = self._X_test_balanced
            y_test = self._y_test_balanced
        else:
            X_test = self._X_test
            y_test = self._y_test

        y_pred = model.predict(X_test)

        model_name = f'{self._model_name}_{"balanced" if balanced_model else "unbalanced"} model_{"balanced" if balanced_test_data else "unbalanced"} test data'

        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred, labels=config.SENTIMENTS))

        print(f"Confusion Matrix for {model_name}:")
        cm = confusion_matrix(y_test, y_pred, labels=config.SENTIMENTS, normalize="true")
        print(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.SENTIMENTS)
        disp.plot(cmap=plt.cm.Blues)

        plt.title(f"Normalized Confusion Matrix for {model_name}")
        plt.show()

    def predict(self, X, balanced_model):
        """_summary_"""
        if balanced_model:
            model = self.pipeline_balanced
        else:
            model = self.pipeline

        return model.predict(X)

    def dump(self, path, name=None):
        """_summary_"""
        if name is None:
            name = self._model_name

        joblib.dump(self, f"{path}/{name}.joblib")


class OllamaPipeline(CustomPipeline):
    def __init__(self, df, features, target, ollama_model, model_name=None):
        steps = []
        super().__init__(df, features, target, steps, model_name)

        self.ollama_model = ollama_model

        ollama.pull(self.ollama_model)
        self.system_instructions = (
            "You are a helpful assistant specialized in analyzing news articles. "
            "Provide concise and accurate responses in JSON format."
        )

    def fit(self):
        """_summary_"""
        raise NotImplementedError("OllamaPipeline: No training required for this pipeline.")

    def evaluate(self):
        model_name = self._model_name

        X_test = self._X_test
        y_test = self._y_test
        y_pred = self.predict(X_test)
        y_pred = pd.Series(y_pred, index=y_test.index)

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
        errors = []
        y_pred = []
        
        for index, row in tqdm.tqdm(X.iterrows(), desc="Analyzing sentiment with Ollama"):
            try:
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