import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from translate import Translator
import pdfplumber

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def translate_text(text, target_language='en'):
    translator = Translator(to_lang=target_language, from_lang='mk')
    return translator.translate(text)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def analyze_sentiment(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    translated_text = translate_text(text)
    sentiment = sia.polarity_scores(translated_text)
    return sentiment

sentiment = analyze_sentiment('kmb_report.pdf')

print("Sentiment Analysis Result:\n", sentiment)


