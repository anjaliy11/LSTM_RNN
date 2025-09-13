# LSTM\_RNN: Shakespearean Text Generation with LSTM

## Overview

LSTM\_RNN is a deep learning project that employs Long Short-Term Memory (LSTM) networks to generate text in the style of Shakespeare. Trained on the hamlet dataset(text of William Shakespeare), this model captures the nuances of Early Modern English to produce coherent and stylistically consistent text.

## Features

* **Character-Level Text Generation**: Generates text one character at a time, predicting the next character based on the previous sequence.
* **Trained on Shakespeare's Works**: Utilizes the dataset from \[The Tragedie of Hamlet by William Shakespeare 1599] to learn the intricacies of Shakespearean language.
* **Streamlit Interface**: Provides an interactive web application for real-time text generation.

## Requirements

To run this project, ensure you have the following Python packages installed:

* `tensorflow`
* `streamlit`
* `numpy`
* `pickle`

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Setup

Clone the repository:

```bash
git clone https://github.com/anjaliy11/LSTM_RNN.git
cd LSTM_RNN
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the Streamlit application:

```bash
streamlit run app.py
```

This command will start a local server, and you can interact with the model through your web browser.

## Model Details

* **Model Architecture**: LSTM-based neural network.
* **Training Data**: hamlet.txt
* **Training Process**: The model was trained to predict the next character in a sequence, capturing the stylistic elements of Shakespearean text.

## Contributing

Contributions are welcome! If you have suggestions for improvements or enhancements, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or feedback, please contact [anjaliy11](https://github.com/anjaliy11).
