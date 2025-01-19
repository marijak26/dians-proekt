import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from translate import Translator
import pdfplumber
import pytesseract
from flask import Flask, request, jsonify

app = Flask(__name__)

class SentimentService:
    def __init__(self):
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()

    def translate_text(self, text):
        translator = Translator(to_lang='en', from_lang='mk')
        chunks = self.chunk_text(text)

        translated_chunks = []
        for chunk in chunks:
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)

        translated_text = " ".join(translated_chunks)
        return translated_text

    def chunk_text(self, text, max_length=500):
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    def extract_text_from_pdf(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def extract_text_from_image_pdf(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                image = page.to_image()
                text += pytesseract.image_to_string(image.original, lang='mkd')
        return text

    def analyze_sentiment(self, pdf_path):
        try:
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                text = self.extract_text_from_image_pdf(pdf_path)

            translated_text = self.translate_text(text)

            sentiment = self.sia.polarity_scores(translated_text)
            compound_score = sentiment['compound']
            sentiment_classification = self.classify_sentiment(compound_score)

            return sentiment_classification

        except FileNotFoundError:
            return "No financial reports from this year found."

    def classify_sentiment(self, compound_score):
        if compound_score > 0.5:
            return "Buy"
        elif 0.2 <= compound_score <= 0.5:
            return "Buy/Hold"
        elif 0.0 <= compound_score < 0.2:
            return "Hold"
        elif -0.2 <= compound_score < 0.0:
            return "Sell"
        else:
            return "Sell/Avoid"


sentiment_service = SentimentService()


@app.route('/api/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    file = request.get_json()['file']
    try:
        sentiment = sentiment_service.analyze_sentiment(file)
        return jsonify({"sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5003, debug=True, host='0.0.0.0')