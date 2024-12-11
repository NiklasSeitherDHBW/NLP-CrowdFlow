class NLTK:
    """_summary_
    """
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target

        # Create a list of transformers dynamically
        transformers = [
            (f'nltk_{feature}', Pipeline([
                ('compound_splitter', CompoundWordSplitter()),
                ('tokenizer', NLTKTokenizer()),
                ('tfidf', TfidfVectorizer())
            ]), feature) for feature in features
        ]

        pipeline_sentiment_nltk = Pipeline(
            steps=[
                ('preprocessing', ColumnTransformer(transformers, remainder='drop')),
                ('clf', RandomForestClassifier(random_state=RANDOM_STATE))
            ],
            verbose=True
        )

        self.pipeline = Pipeline(
            steps=[
                ('preprocessing', ColumnTransformer(transformers, remainder='drop')),
                ('clf', RandomForestClassifier(random_state=RANDOM_STATE))
            ], 
            verbose=True
        )


    def fit(self):
        """_summary_
        """
        X_train, y_train, X_test, y_test = train_test_split(self.df[self.features], self.df[self.target], test_size=0.2)
        self.pipeline.fit()
