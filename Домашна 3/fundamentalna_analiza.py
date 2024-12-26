import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from translate import Translator
import pdfplumber
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def translate_text(text):
    translator = Translator(to_lang='en', from_lang='mk')
    chunks = chunk_text(text)

    translated_chunks = []
    for chunk in chunks:
        translated_chunk = translator.translate(chunk)
        translated_chunks.append(translated_chunk)

    translated_text = " ".join(translated_chunks)
    return translated_text


def chunk_text(text, max_length=500):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


def extract_text_from_image_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            image = page.to_image()
            text += pytesseract.image_to_string(image.original, lang='mkd')
    return text


def analyze_sentiment(pdf_path):
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        text = extract_text_from_image_pdf(pdf_path)

    translated_text = translate_text(text)

    sentiment = sia.polarity_scores(translated_text)
    return sentiment


sentiment = analyze_sentiment('adin_report.pdf')

print("Sentiment Analysis Result:\n", sentiment)
