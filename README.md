# 💬 PyTorch Chatbot

A simple rule-based chatbot built using Python and PyTorch. It uses a neural network to classify user intents from input text and responds based on predefined responses in a JSON file.

## 📁 Project Structure

```
.
├── intents.json         # Intent data with patterns and responses
├── model.py             # Neural network model definition (NeuralNet)
├── nltk_utils.py        # Tokenization, stemming, and bag-of-words utilities
├── train.py             # Trains the chatbot model and saves it to data.pth
├── chat.py              # Inference script for chatting with the bot
├── data.pth             # Trained model and metadata
└── README.md            # Project documentation
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pytorch-chatbot.git
cd pytorch-chatbot
```

### 2. Install Requirements

```bash
pip install torch nltk numpy
```

Also run once:

```python
import nltk
nltk.download('punkt')
```

### 3. Train the Model

To train the model and generate `data.pth`:

```bash
python train.py
```

### 4. Chat with the Bot

```bash
python chat.py
```

Type your message and the bot will respond. Type `quit` to exit.

## 📄 intents.json Format

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "How are you?"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    }
  ]
}
```

## 🧠 How It Works

1. **Preprocessing**: Text patterns are tokenized, stemmed, and converted to a bag-of-words vector.
2. **Training**: A feedforward neural network is trained using these vectors and associated tags.
3. **Prediction**: At runtime, user input is preprocessed and passed through the model to predict intent.
4. **Response**: A response is chosen randomly from the corresponding intent.

## 🔧 Customization

- To add new functionality, update `intents.json` with new tags, patterns, and responses.
- Retrain the model using `python train.py`.

## 🧪 Example

```
You: hello
Sam: Hello! How can I assist you?

You: what can you do?
Sam: I can chat with you and help answer simple questions.
```

## 📜 License

This project is open-source under the MIT License.
