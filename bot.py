from aiogram.contrib.middlewares.logging import LoggingMiddleware

import logging
import nst
import os
import state
from state import Test
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from shutil import copyfile
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from settings import (BOT_TOKEN, HEROKU_APP_NAME,
                      WEBHOOK_URL, WEBHOOK_PATH,
                      WEBAPP_HOST, WEBAPP_PORT)

logging.basicConfig(level=logging.INFO)

# initialize Bot
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())


# start message
@dp.message_handler(commands=['start'], state=None)
async def send_welcome(message: types.Message):
    # keyboard
    markup_general = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_general = types.KeyboardButton('Прикрепить фото')

    markup_general.add(button_general)

    await bot.send_message(message.chat.id, 'Привет, {0.first_name}!\nЯ - <b>{1.first_name}</b>, бот, стилизующий'
                                            'фотографии. Давай я превращу твое фото в картину!'.format(
        message.from_user, await bot.get_me()),
                           parse_mode='html', reply_markup=markup_general)
    await bot.send_message(message.chat.id, 'Остались вопросы? ➡️ /help')
    await Test.first()


# help
@dp.message_handler(commands=['help'], state=Test.P1)
async def help_message(message: types.Message):
    await bot.send_message(message.chat.id,
                           'Я умею переносить стиль с одного фото на другое при помощи нейронной сети. Тебе '
                           'лишь нужно отправить мне фото и выбрать один из предложенных стилей."')


# chat
@dp.message_handler(content_types=['text'], state=Test.P1)
async def chat1(message: types.Message):
    if message.chat.type == 'private':
        if message.text == 'Прикрепить фото':
            await bot.send_message(message.chat.id,
                                   'Пожалуйста, отправь мне фото, которое хочешь преобразовать. '
                                   'Рекомендуется, чтобы объект на фото располагался по центру, так как в процессе '
                                   'обработки фото кадрируется.')
        else:
            await bot.send_message(message.chat.id, 'Я не знаю, что ответить :(')


# save photo

@dp.message_handler(content_types=['photo'], state=Test.P1)
async def handle_photo1(message: types.Message, state: FSMContext):
    await message.photo[-1].download('./images/styles/content_image.jpg')
    await bot.send_message(message.chat.id, 'Фото успешно загружено!')
    await bot.send_message(message.chat.id, 'А теперь выбери стиль ')
    style_reply_markup = types.InlineKeyboardMarkup(row_width=2, one_time_keyboard=True)
    style_reply_button1 = types.InlineKeyboardButton('Сидящая обнаженная', callback_data='picasso')
    style_reply_button2 = types.InlineKeyboardButton('Звездная ночь', callback_data='van_gogh')
    style_reply_button3 = types.InlineKeyboardButton('Современное исскуство', callback_data='current')
    style_reply_button4 = types.InlineKeyboardButton('Свое фото стиля', callback_data='svoi')

    style_reply_markup.add(style_reply_button1, style_reply_button2, style_reply_button3, style_reply_button4)

    media = types.MediaGroup()
    media.attach_photo(types.InputFile('images/styles/picasso.jpg'), 'Сидящая Обнаженная')
    media.attach_photo(types.InputFile('images/styles/van_gogh.jpg'), 'Звездная ночь')
    media.attach_photo(types.InputFile('images/styles/current.jpg'), 'Современное исскуство')

    await bot.send_media_group(chat_id=message.chat.id, media=media)
    await bot.send_message(message.chat.id, 'Мои варианты: ', reply_markup=style_reply_markup)


@dp.message_handler(content_types=['photo'], state=Test.P2)
async def handle_photo2(message: types.Message, state: FSMContext):
    await message.photo[-1].download('./images/styles/mystyle_image.jpg')
    await bot.send_message(message.chat.id, 'Получил')
    copyfile('images/styles/mystyle_image.jpg', 'images/styles/style_image.jpg')
    await state.finish()
    await launch_nst(message)


# callback
@dp.callback_query_handler(lambda call: True, state=Test.P1)
async def callback_inline1(call):
    try:
        if call.message:
            if call.data == 'van_gogh':
                await bot.send_message(call.message.chat.id, 'Я тоже люблю импрессионизм! ')
                copyfile('images/styles/van_gogh.jpg', 'images/styles/style_image.jpg')
                await launch_nst(call.message)
            elif call.data == 'picasso':
                await bot.send_message(call.message.chat.id, 'Прекрасный выбор!')
                copyfile('images/styles/picasso.jpg', 'images/styles/style_image.jpg')
                await launch_nst(call.message)
            elif call.data == 'current':
                await bot.send_message(call.message.chat.id, 'Глубоко')
                copyfile('images/styles/current.jpg', 'images/styles/style_image.jpg')
                await launch_nst(call.message)
            elif call.data == 'svoi':
                await bot.send_message(call.message.chat.id,
                                       'Интересно поработать с чем-то новым! Cкинь мне свое фото стиля')

            # remove inline buttons
            await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text='...',
                                        reply_markup=None)
            await Test.next()

    except Exception as e:
        print(repr(e))


async def launch_nst(message):
    # print('ok')
    content_image_name = 'images/styles/content_image.jpg'
    style_image_name = 'images/styles/style_image.jpg'
    content = open(content_image_name, 'rb')
    style = open(style_image_name, 'rb')

    await bot.send_message(message.chat.id, 'Начинаю обработку. Перенос стиля займет примерно 15 минут ')

    await nst.main(content_image_name, style_image_name)

    await bot.send_message(message.chat.id, 'Готово!')
    result = open('images/result/result.jpg', 'rb')
    await bot.send_photo(message.chat.id, result)

    result_reply_markup = types.InlineKeyboardMarkup(row_width=2)
    result_reply_button1 = types.InlineKeyboardButton('Вау, шикарно!', callback_data='amazing')
    result_reply_button2 = types.InlineKeyboardButton('Хорошо', callback_data='nice')
    result_reply_button3 = types.InlineKeyboardButton('Нормальвно', callback_data='ok')
    result_reply_button4 = types.InlineKeyboardButton('Ой...кошмар', callback_data='gross')

    result_reply_markup.add(result_reply_button1, result_reply_button2, result_reply_button3, result_reply_button4)

    await bot.send_message(message.chat.id, 'Ну, как тебе?', reply_markup=result_reply_markup


@dp.callback_query_handler(lambda call: True, state=Test.P2)
async def callback_inline2(call):
    try:
        if call.message:
            # result feedback
            if call.data == 'amazing':
                await bot.send_message(call.message.chat.id, 'Рад что тебе понравилось!')
            elif call.data == 'nice':
                await bot.send_message(call.message.chat.id, 'Супер!')
            elif call.data == 'ok':
                await bot.send_message(call.message.chat.id, 'Может, в следующий раз у меня лучше получится. 😕')
            elif call.data == 'gross':
                await bot.send_message(call.message.chat.id, 'Надеюсь, это останется между нами... 👉👈')

            # remove inline buttons
            await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text='...',
                                        reply_markup=None)
            await Test.first()

            await bot.send_message(call.message.chat.id, 'Хочешь еще раз? Просто отправь мне фото, которое хочешь '
                                                         'стилизовать')

    except Exception as e:
        print(repr(e))


@dp.callback_query_handler(lambda call: True, state=None)
async def callback_inline3(call):
    try:
        if call.message:
            # result feedback
            if call.data == 'amazing':
                await bot.send_message(call.message.chat.id, 'Рад что тебе понравилось!')
            elif call.data == 'nice':
                await bot.send_message(call.message.chat.id, 'Супер!')
            elif call.data == 'ok':
                await bot.send_message(call.message.chat.id, 'Может, в следующий раз у меня лучше получится. 😕')
            elif call.data == 'gross':
                await bot.send_message(call.message.chat.id, 'Надеюсь, это останется между нами... 👉👈')

            # remove inline buttons
            await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text='...',
                                        reply_markup=None)
            await Test.first()

            await bot.send_message(call.message.chat.id, 'Хочешь еще раз? Просто отправь мне фото, которое хочешь '
                                                         'стилизовать')

    except Exception as e:
        print(repr(e))


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
