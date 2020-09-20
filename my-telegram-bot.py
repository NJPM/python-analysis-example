from telegram.ext import Updater, CommandHandler
import requests
import re

def get_url():
    contents = requests.get('https://random.dog/woof.json').json()
    url = contents['url']
    return url

def bop(bot, update):
    url = get_url()
    chat_id = update.message.chat_id
    bot.send_photo(chat_id = chat_id, photo = url)

def main():
    updater = Updater('1184795691:AAGv9VpEJf1Ri305WYMnXJkni9LIkthT66U', use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('bop',bop))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()


# 1184795691:AAGv9VpEJf1Ri305WYMnXJkni9LIkthT66U
