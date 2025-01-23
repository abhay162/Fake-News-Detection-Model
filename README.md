# Fake News Detection

This project uses machine learning techniques to detect fake news articles. The dataset consists of labeled articles, and the model classifies them as either "real" or "fake."

## Files

- `train_model.py`: Script to train the machine learning model.
- `predict.py`: Script to predict whether a new article is fake or real.
- `data_preprocessing.ipynb`: Jupyter notebook for preprocessing the text data.
- `requirements.txt`: List of dependencies.

## Setup

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/fake-news-detection.git
    cd fake-news-detection
    ```

2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Preprocess your dataset (if not already done):

    Open `data_preprocessing.ipynb` in Jupyter Notebook, and run the preprocessing steps.

4. Train the model:

    ```
    python train_model.py
    ```

5. Make predictions:

    ```
    python predict.py --input "Sample news article text"
    ```

## License

This project is licensed under the MIT License.
