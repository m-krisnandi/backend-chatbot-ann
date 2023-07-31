import os
from dotenv import load_dotenv
from flask import Flask
from chat import get_response
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

app = Flask(__name__)
load_dotenv()

# replace with your Telegram API token
bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
bot = telegram.Bot(token=bot_token)

# Fungsi handler untuk command /start
def start(update, context):
    chat_id = update.message.chat_id
    context.bot.send_message(
        chat_id=chat_id, text="Halo. Selamat datang di chatbot informasi objek wisata di Kabupaten Bandung. Silahkan ketikkan pertanyaan anda.")


def echo(update, context):
    """
        Fungsi ini merespons pesan yang diterima dari pengguna.

        Parameter:
        - update: Objek yang berisi data terkait pesan yang diterima oleh bot.
        - context: Objek yang menyimpan konteks dari pesan yang diterima, dan digunakan untuk berkomunikasi dengan Telegram API.
    """
    chat_id = update.message.chat_id
    text = update.message.text

    # Memeriksa apakah teks yang diterima valid dengan menggunakan fungsi get_response
    chat_response = get_response(text)

    # Mengirimkan respons teks ke Telegram (default)
    response_message = context.bot.send_message(chat_id=chat_id, text=f"{chat_response.response}",
                                                parse_mode=telegram.ParseMode.MARKDOWN)

    # Memeriksa apakah terdapat image_url
    if chat_response.image_url:
        # Jika ada URL gambar, kirimkan gambar ke pengguna
        context.bot.send_photo(chat_id=chat_id, photo=chat_response.image_url)
        # Ubah pesan untuk memberitahu bahwa gambar telah terkirim
        response_message.edit_text(f"{chat_response.response}\n\n*Gambar terkirim!*",
                                   parse_mode=telegram.ParseMode.MARKDOWN)

    # Memeriksa apakah terdapat koordinat
    if chat_response.coordinates:
        # Jika ada koordinat, tampilkan lokasi di peta
        lat, lon = chat_response.coordinates.split(',')
        context.bot.send_message(chat_id=chat_id, text=f"*Lokasi di Peta:*", parse_mode=telegram.ParseMode.MARKDOWN)
        context.bot.send_location(chat_id=chat_id, latitude=float(lat), longitude=float(lon))


# Fungsi untuk mengecek bot hidup
@app.route('/ping')
def ping():
    return "Bot is alive!"

# Membuat objek Updater dengan menggunakan token bot dan mode context
updater = Updater(bot_token, use_context=True)

# Tambahkan handler untuk command /start
updater.dispatcher.add_handler(CommandHandler('start', start))
# Tambahkan handler untuk echo/respons teks
updater.dispatcher.add_handler(MessageHandler(Filters.text, echo))

# jalankan webhook production mode tes menggunakan ngrok
updater.start_webhook(listen="0.0.0.0",
                      port=int(os.environ.get('PORT', 443)),
                      url_path=bot_token,
                      webhook_url='https://95bd-103-147-9-232.ngrok-free.app/' + bot_token)

# jalankan webhook production mode menggunakan cloud run
# updater.start_webhook(listen="0.0.0.0",
#                       port=int(os.environ.get('PORT', 443)),
#                       url_path=bot_token,
#                       webhook_url='https://backend-chatboot-ann-5j2ubmycza-et.a.run.app/' + bot_token)

# Hentikan webhook yang sedang berjalan
# updater.bot.delete_webhook()

# Jalankan bot dalam mode polling (untuk development)
# updater.start_polling()
# updater.idle()

if __name__ == '__main__':
    app.run(debug=True)
