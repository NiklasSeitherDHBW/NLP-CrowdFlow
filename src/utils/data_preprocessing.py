"""_summary_
"""

import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
import src.utils.config as Config

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
            df = df[df[target] == sentiment]
            if count < max_count:
                df_resampled = resample(
                    df, replace=True, n_samples=max_count, random_state=Config.RANDOM_STATE
                )
                balanced_dfs.append(df_resampled)
            else:
                balanced_dfs.append(df)

        df_balanced = pd.concat(balanced_dfs)

        return df_balanced


class NLTKTokenizer(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("punkt_tab")
        self.stop_words = set(stopwords.words('english'))

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
        tokens = word_tokenize(text.lower())
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
