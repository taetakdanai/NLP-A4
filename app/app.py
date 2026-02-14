from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

# Initialize the Flask app
app = Flask(__name__)

# Update this to the correct path containing model weights and tokenizer
model_path = r'D:\VS_Code\NLP\NLP-A4\artifacts\bert_mlm_ckpt\checkpoint-3500'

# Load the custom-trained Sentence-BERT model from the checkpoint
model = SentenceTransformer(model_path)

# Define a function to predict NLI
def predict_nli(premise, hypothesis):
    # Encode the sentences
    premise_embedding = model.encode(premise, convert_to_tensor=True)
    hypothesis_embedding = model.encode(hypothesis, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_similarity = util.pytorch_cos_sim(premise_embedding, hypothesis_embedding)

    # Simple threshold to categorize NLI (you can adjust this based on your model's outputs)
    if cosine_similarity > 0.8:
        return "Entailment"
    elif cosine_similarity > 0.4:
        return "Neutral"
    else:
        return "Contradiction"

# Define the routes for the app
@app.route('/', methods=['GET', 'POST'])
def index():
    label = ""
    if request.method == 'POST':
        premise = request.form['premise']
        hypothesis = request.form['hypothesis']
        label = predict_nli(premise, hypothesis)
    return render_template('index.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)
