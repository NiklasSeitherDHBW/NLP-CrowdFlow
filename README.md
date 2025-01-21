## Prerequisites
Dieses Projekt wurde in einer Python-3.11.10-Umgebung entwickelt. Die benötigten Pakete können mit folgendem Befehl installiert werden:
```bash
pip install -r "requirements.txt
```

Zusätzlich wird ein vortrainiertes Word2Vec-Modell genutzt dass über folgenden Link heruntergladen werden kann: [GoogleNews-vectors-negative300](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors/data). Dieses muss nach dem Download in den Ordner `res/models` abgelegt werden.

Aufgrund der Dateigröße wurden lediglich die unverarbeiteten Datenästze im Ordner `res/input` abgelegt. Zum Teil werden die Datensätze aber auch erst zur Laufzeit (von Huggingface) heruntergalden und die verarbeitete Version abgespeichert. Bevor die Modelle trainiert und in der Anwendungsoberfläche verwendet werden können, muss das `src/data_preparation.ipynb` Notebook ausgeführt werden. Dieses erstellt die fertig prozssierten Datensätze und legt sie im Ordner `res/prepared` ab. Ist dieser Schritt abgeschlossen, kann in der Datei `src/utils/config.py` der gewünschte Trainings- sowie Cross-Validierungsdatensatz angegeben werden. Hiernach können die verschiedenen Notebooks zum Trainieren und Evaluieren der Modelle ausgeführt werden.

## Use Case

Analyzing the sentiment in blog posts offers investors additional valuable insights into the market and the people at a particular time. Since crypto currencies don't have an intrinsic value connected to the reality opposed to assets like company stocks the peoples sentiment can have drastic influence on their performance.

To achieve visual analytics of the connection of the current sentiment towards specific crypto currencies and the peoples current sentiment towards them we build an application called CrowdFlow which retrieves the financial data as well as relevant news articles which are classified based on their sentiment.

## Application Architecture

The application is divided into the two components visualized in the figure below. 

The first one is the frontend delivering the visual representation of a crypto currencies performance and trading volume as well as the time-related sentiment analysis. The frontend offers the possibility for choosing the desired cryptocurrency as well as an individual period of time. The currencies perfomance in that time frame is displayed with a candle diagram, while the traded volume can be found in a bar chart below the performance graph. The time-related sentiment is displayed below the financial information using a stacked bar chart. For deeper analysis of the sentiment, the analized text instance are labeled and displayed and the right side of the frontend.

This data is retrieved and created by the backend consisting of corresponding API calls to Yahoo Finance for both the financial data as well as the last 200 crypto related news. While the financial data is simmply displayed, the news are delivered as input for a sklearn pipeline including the preprocessing steps and the model prediction.

<img src="https://raw.githubusercontent.com/NiklasSeitherDHBW/NLP-CrowdFlow/refs/heads/main/docs/NLP_Applikation.svg" alt="Application Architecture" width="50%">


