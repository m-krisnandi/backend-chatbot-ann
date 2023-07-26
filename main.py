import os
from dotenv import load_dotenv
from threading import Thread
from flask import Flask
from chat import get_response
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

app = Flask(__name__)
load_dotenv()

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
    chat_response = get_response(text)

    # send response to Telegram
    if chat_response.image_url:
        response_message = context.bot.send_message(
            chat_id=chat_id, text=f"{chat_response.response}\n\n*Mengirim gambar...*", parse_mode=telegram.ParseMode.MARKDOWN)
        
        # Dapatkan URL publik gambar yang diunggah
        image_public_url = chat_response.image_url
        
        context.bot.send_photo(chat_id=chat_id, photo=image_public_url)
        response_message.edit_text(f"{chat_response.response}\n\n*Gambar terkirim!*", parse_mode=telegram.ParseMode.MARKDOWN)

         # Tampilkan peta jika koordinat tersedia
        if chat_response.coordinates:
            lat, lon = chat_response.coordinates.split(',')
            context.bot.send_message(chat_id=chat_id, text=f"*Lokasi di Peta:*", parse_mode=telegram.ParseMode.MARKDOWN)
            context.bot.send_location(chat_id=chat_id, latitude=float(lat), longitude=float(lon))
    else:
        context.bot.send_message(chat_id=chat_id, text=f"{chat_response.response}", parse_mode=telegram.ParseMode.MARKDOWN)


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
