"""_summary_
"""

import copy
import re
import string
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import nltk
from . import config


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
