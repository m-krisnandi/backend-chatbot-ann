
# Chatbot Objek Wisata di Kabupaten Bandung

Backend for Chatbot Objek Wisata di Kabupaten Bandung dengan menggunakan Algoritma Artificial Neural Network dan terintegrasi dengan Telegram dan deployment dataset di Redis dan backend di Cloud Run GCP


## Installation

Clone the project

```bash
  git clone -b no-softmax https://github.com/m-krisnandi/backend-chatbot-ann.git
```

Go to the project directory

```bash
  cd backend-chatbot-ann
```

Install dependencies

```bash
  pip install --no-cache-dir -r requirements.txt
```
Create API Bot Telegram using BotFather

https://telegram.me/BotFather

Tutorial create bot using BotFather

- /start

- /newbot

- create bot name

- Telegram Bot API token obtained





## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`FLASK_APP=main.py`

`TELEGRAM_BOT_TOKEN`

`REDIS_HOST`

`REDIS_PORT`

`REDIS_PASSWORD`


## Run Locally

Run backend-chatbot-ann with flask

```bash
  flask run
```
    
## Testing Bot Telegram

To test a Telegram bot, you can search for a pre-made bot created with BotFather and try sending text related to tourist attractions in the Bandung district.

