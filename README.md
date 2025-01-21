## Prerequisites
This project was developed in a **Python 3.11.10** environment. Follow the steps below to set up the project and prepare the data:

### 1. Install Dependencies

Install the required Python packages using the following command:
```bash
pip install -r "requirements.txt"
```

### 2. Download the Pretrained Word2Vec Model

A pretrained Word2Vec model is required for this project. You can obtain it from the following link: [GoogleNews-vectors-negative300](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors/data). After downloading, place the file in the folder:
```
res/models
```
> **Note**: If the `res/models` folder does not exist, create it manually.

### 3. Dataset Preparation

The project relies on datasets stored in the following folder structure:
```
res/input        # Raw datasets
res/prepared     # Processed datasets (generated during preparation)
```

#### Steps for Dataset Preparation:
1. **Raw datasets**:
   - Some datasets are downloaded automatically at runtime (e.g., from Huggingface). Others are already available in the `res/input` folder.

2. **Data Preparation Notebook**:
   - Run the notebook located at:
     ```
     src/data_preparation.ipynb
     ```
   - This notebook processes the raw datasets and generates ready-to-use datasets. These processed datasets will be saved in the `res/prepared` folder.

### 4. Configure Training and Cross-Validation

Once the data preparation is complete, specify the desired training and cross-validation settings in the following configuration file:
```
src/utils/config.py
```

### 5. Model Training

After preparing the datasets and updating the configuration file, you can execute the provided notebooks to train the models. Ensure you have sufficient computational resources for this step as both, the ollama and the Word2Vec model rely heavily on computational resources.


## Use Case

Analyzing the sentiment in blog posts offers investors additional valuable insights into the market and the people at a particular time. Since crypto currencies don't have an intrinsic value connected to the reality opposed to assets like company stocks the peoples sentiment can have drastic influence on their performance.

To achieve visual analytics of the connection of the current sentiment towards specific crypto currencies and the peoples current sentiment towards them we build an application called CrowdFlow which retrieves the financial data as well as relevant news articles which are classified based on their sentiment.

## Application Architecture

The application is divided into the two components visualized in the figure below. 

The first one is the frontend delivering the visual representation of a crypto currencies performance and trading volume as well as the time-related sentiment analysis. The frontend offers the possibility for choosing the desired cryptocurrency as well as an individual period of time. The currencies perfomance in that time frame is displayed with a candle diagram, while the traded volume can be found in a bar chart below the performance graph. The time-related sentiment is displayed below the financial information using a stacked bar chart. For deeper analysis of the sentiment, the analized text instance are labeled and displayed and the right side of the frontend.

This data is retrieved and created by the backend consisting of corresponding API calls to Yahoo Finance for both the financial data as well as the last 200 crypto related news. While the financial data is simmply displayed, the news are delivered as input for a sklearn pipeline including the preprocessing steps and the model prediction.

A more thorough description of the model training and its methodologie as well as the data used can be found in the file [Documentation.pdf](/tree/main/docs/documentation.pdf).

<img src="https://raw.githubusercontent.com/NiklasSeitherDHBW/NLP-CrowdFlow/refs/heads/main/docs/NLP_Applikation.svg" alt="Application Architecture" width="50%">


