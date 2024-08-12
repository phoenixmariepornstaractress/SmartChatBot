import os
import telebot
import logging
import time
import schedule
from datetime import datetime
from telebot.types import Update
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect
from PIL import Image
import requests
from io import BytesIO

# Load bot token from environment variable for better security
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("Please set the BOT_TOKEN environment variable.")

# Initialize the bot
bot = telebot.TeleBot(BOT_TOKEN)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load AI/NLP pipelines
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
translation_model = pipeline("translation_en_to_fr")
generator = pipeline("text-generation", model="gpt-2")
image_captioner = pipeline("image-captioning", model="nlpconnect/vit-gpt2-image-captioning")

# Advanced translation models
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translation_model_zh = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

def get_latest_chat_info():
    try:
        updates = bot.get_updates(timeout=10)
        if updates:
            latest_update = updates[-1]
            chat = latest_update.message.chat
            message_text = latest_update.message.text

            chat_id = chat.id
            chat_title = chat.title or chat.username or "Private Chat"
            chat_type = chat.type

            logging.info(f"Chat ID: {chat_id}, Chat Title: {chat_title}, Chat Type: {chat_type}, Message Text: {message_text}")

            bot.get_updates(offset=latest_update.update_id + 1)

            return chat_id, chat_title, chat_type, message_text

    except telebot.apihelper.ApiTelegramException as api_ex:
        logging.error(f"Telegram API error: {api_ex}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return None, None, None, None

def send_message(chat_id, message):
    try:
        bot.send_message(chat_id, message)
        logging.info(f"Message sent successfully to chat ID: {chat_id}")
    except telebot.apihelper.ApiTelegramException as api_ex:
        logging.error(f"Telegram API error occurred while sending message: {api_ex}")
    except Exception as e:
        logging.error(f"Failed to send message: {e}")

def send_document():
    chat_id, _, _, _ = get_latest_chat_info()

    if chat_id:
        document_path = "path/to/your/document.pdf"  # Replace with your document path

        try:
            with open(document_path, 'r') as document:
                document_content = document.read()
                summary = summarizer(document_content, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                
                bot.send_message(chat_id, f"Document Summary: {summary}")
                document.seek(0)  # Reset the file pointer to the beginning
                bot.send_document(chat_id, document)

            logging.info(f"Document sent successfully to chat ID: {chat_id}")
        except Exception as e:
            logging.error(f"Failed to send document: {e}")
    else:
        logging.warning("No valid chat information retrieved for sending scheduled document.")

def analyze_sentiment(message_text):
    sentiment_result = sentiment_analyzer(message_text)
    return sentiment_result[0]['label'].lower()

def classify_message(message_text):
    candidate_labels = ["greeting", "question", "complaint", "feedback", "other"]
    classification = classifier(message_text, candidate_labels)
    return classification['labels'][0]  # Return the most likely label

def detect_language(message_text):
    return detect(message_text)

def translate_message(message_text, target_language="fr"):
    if target_language == "zh":
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        inputs = tokenizer.encode(message_text, return_tensors="pt", padding=True)
        outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        translation = translation_model(message_text)
        return translation[0]['translation_text']

def generate_response(prompt):
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

def caption_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    caption = image_captioner(img)
    return caption[0]['generated_text']

def text():
    chat_id, _, chat_type, message_text = get_latest_chat_info()

    if chat_id:
        sentiment = analyze_sentiment(message_text)
        classification = classify_message(message_text)
        detected_language = detect_language(message_text)
        translated_message = translate_message(message_text) if detected_language == "en" else "Not translated"
        generated_response = generate_response(message_text)
        
        message = (
            f"Hello! I detected that your message sentiment is {sentiment}, "
            f"and it seems to be a {classification}. "
            f"Detected language: {detected_language}. "
            f"Here’s the French translation: {translated_message}. "
            f"AI-generated response: {generated_response}"
        )
        
        send_message(chat_id, message)
    else:
        logging.warning("No valid chat information retrieved for sending scheduled message.")

def handle_new_message():
    chat_id, chat_title, chat_type, message_text = get_latest_chat_info()

    if chat_id:
        sentiment = analyze_sentiment(message_text)
        classification = classify_message(message_text)
        detected_language = detect_language(message_text)
        translated_message = translate_message(message_text, "zh") if detected_language == "en" else "Not translated"
        generated_response = generate_response(message_text)
        
        response_message = (
            f"Hello, {'group' if chat_type != 'private' else 'user'}! "
            f"I detected that your message sentiment is {sentiment}, "
            f"and it seems to be a {classification}. "
            f"Detected language: {detected_language}. "
            f"Here’s the Chinese translation: {translated_message}. "
            f"AI-generated response: {generated_response}"
        )
        send_message(chat_id, response_message)
    else:
        logging.warning("No valid chat information retrieved. Message not sent.")

def handle_image_message(chat_id, image_url):
    caption = caption_image(image_url)
    send_message(chat_id, f"Image caption: {caption}")

def handle_voice_message(chat_id, file_id):
    # You would integrate a speech-to-text model here
    # Example: Use OpenAI's Whisper or similar
    pass

def schedule_daily_tasks():
    schedule.every().day.at("10:00").do(text)
    schedule.every().day.at("21:00").do(send_document)

def run_bot():
    logging.info("Bot is running...")

    schedule_daily_tasks()

    try:
        while True:
            try:
                bot.polling(none_stop=True, interval=0)
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"An error occurred during polling: {e}")
                bot.stop_polling()
                time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
        bot.stop_polling()

if __name__ == "__main__":
    run_bot()
