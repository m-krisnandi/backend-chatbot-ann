import os
from flask_cors import CORS
from dotenv import load_dotenv
from threading import Thread
from flask import Flask, request, jsonify
from chat import get_response
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

app = Flask(__name__)
load_dotenv()
CORS(app)

# replace with your Telegram API token
bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
bot = telegram.Bot(token=bot_token)

def start(update, context):
    chat_id = update.message.chat_id
    context.bot.send_message(
        chat_id=chat_id, text="Halo. Selamat datang di chatbot informasi objek wisata di Kabupaten Bandung. Silahkan ketikkan pertanyaan anda.")


def echo(update, context):
    chat_id = update.message.chat_id
    text = update.message.text

    # check if text is valid
    response, image_url = get_response(text)

    # send response to Telegram
    if image_url:
        response_message = context.bot.send_message(
            chat_id=chat_id, text=response + "\n\n*Mengirim gambar...* ", parse_mode=telegram.ParseMode.MARKDOWN)
        
        # Dapatkan URL publik gambar yang diunggah
        image_public_url = image_url
        
        context.bot.send_photo(chat_id=chat_id, photo=image_public_url)
        response_message.edit_text(response + "\n\n*Gambar terkirim!*", parse_mode=telegram.ParseMode.MARKDOWN)
    else:
        context.bot.send_message(chat_id=chat_id, text=response, parse_mode=telegram.ParseMode.MARKDOWN)


updater = Updater(token=bot_token, use_context=True)

updater.dispatcher.add_handler(CommandHandler("start", start))
updater.dispatcher.add_handler(MessageHandler(
    Filters.text & ~Filters.command, echo))


def start_polling():
    updater.start_polling()


thread = Thread(target=start_polling)
thread.start()


@app.route('/')
def home():
    return "Server is running..."

if __name__ == '__main__':
    updater.start_polling()
    app.run(debug=True)
