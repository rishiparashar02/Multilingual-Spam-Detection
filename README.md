📡 Multilingual Spam Detection (English / German / French)

This project builds a multilingual SMS spam classifier using BERT, trained on a custom dataset containing messages in English, German, and French.
The model predicts whether a message is spam or ham across multiple languages.

🚀 Features

Multilingual dataset (EN, DE, FR)

Preprocessing + EDA

BERT-based text classification using HuggingFace Transformers

Train/validation split

Model evaluation (accuracy, classification report)

Saved trained model for reuse

📁 Project Structure
├── sms_spam_detection.ipynb   # Main ML notebook
├── (https://www.kaggle.com/datasets/rajnathpatel/multilingual-spam-data) # Drop Hindi Column
├── README.md                   # Project documentation
└── .gitignore                 # To exclude unnecessary files

📊 Dataset

Columns used:

text → English/German/French SMS message

labels → ham or spam

The notebook automatically:

Cleans text

Removes duplicates

Computes text statistics

Visualizes label distribution

🧠 Model Training (BERT)

This project uses:

bert-base-multilingual-cased


The model handles 100+ languages, making it ideal for multilingual spam detection.

Training steps include:

Tokenization

Creating DataLoaders

Fine-tuning BERT

Evaluation & metrics

📦 Install Requirements
pip install torch torchvision torchaudio transformers nltk pandas numpy scikit-learn matplotlib

▶️ Running the Notebook

Open the notebook:

sms_spam_detection.ipynb


and run all cells in order.

📈 Results

The model outputs:

Accuracy

Precision / Recall / F1

Classification report

Loss curves (optional)

💾 Saving the Model

The notebook includes:

torch.save(model.state_dict(), "spam_model.pt")

👨‍💻 Author

Rishi Parashar