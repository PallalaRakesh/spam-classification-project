# 📧 Spam Detection with Neural Networks

This project demonstrates how to build and train a neural network to classify text messages as either "spam" or "ham" (not spam). The process involves cleaning and preparing text data, building a simple deep learning model, and evaluating its performance.

## ⚙️ Project Setup

First, you need to install the necessary libraries. This project uses **TensorFlow**, **scikit-learn**, **pandas**, and **nltk**.

```bash
pip install tensorflow scikit-learn pandas nltk
```

After installing `nltk`, you must download its data packages.

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

This prepares our tools for text processing. `stopwords` are common words like "the" or "is," `punkt` helps with tokenization, and `wordnet` is a dictionary used for finding the base form of words (lemmatization).

## 📋 Dataset

The model is trained on a CSV file named `Spam-Classification.csv`, which contains two columns:

  * **`SMS`**: The text content of the message.
  * **`CLASS`**: The label, either "spam" or "ham."

## 📝 Step-by-Step Explanation

### 1\. Data Preprocessing

The raw text data must be converted into a numerical format that a neural network can understand. This involves several steps:

  * **Tokenization**: We split each message into individual words.
  * **Cleaning**: We remove common **stopwords** and punctuation.
  * **Lemmatization**: We convert words to their base form (e.g., "running" becomes "run").
  * **Vectorization**: We use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the cleaned text into a numerical matrix. TF-IDF gives more importance to words that are frequent in a specific message but rare across all messages, helping the model identify key spam words.

### 2\. Model Architecture

The model is a simple **sequential neural network** built with Keras. It consists of three layers:

  * **Input Layer**: A `Dense` (fully connected) layer that takes the TF-IDF vector as input. It has 32 neurons and uses a **ReLU** activation function.
  * **Hidden Layer**: Another `Dense` layer with 32 neurons, also using **ReLU**. This layer helps the network learn more complex patterns.
  * **Output Layer**: A final `Dense` layer with 2 neurons (one for "ham" and one for "spam"). It uses a **softmax** activation function, which provides a probability score for each class.

### 3\. Training and Evaluation

The model is trained on a portion of the data and then tested on unseen data to ensure it can generalize well.

  * The data is split into **training (90%)** and **testing (10%)** sets.
  * The model is trained over **10 epochs** in batches of 256.
  * During training, we monitor the **`accuracy`** metric.
  * After training, the model's final performance is evaluated on the test set, providing a realistic measure of its effectiveness.

### 4\. Making Predictions

Finally, we use the trained model to predict whether new messages are spam or not. The new message is first converted into a TF-IDF vector using the same vectorizer, and then fed into the neural network for classification.

## 🚀 How to Run the Project

1.  Clone this repository to your local machine.
2.  Install the required libraries.
3.  Place your `Spam-Classification.csv` file in the project directory.
4.  Run the Jupyter Notebook or Python script to execute the code.

By following these steps, you can train your own spam detection model and test its performance\!
