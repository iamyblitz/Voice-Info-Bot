import os
import shutil
import telebot
import requests
import speech_recognition as sr
import subprocess
import datetime
from keras.models import model_from_json
import os
from Func_Transformation_voice_spectogramm import generate_spectogram
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
import numpy as np

token = '1840603958:AAHYjyuGKnfIFlEmclpx-9agQywzI3Cpj-I'

bot = telebot.TeleBot("1840603958:AAHYjyuGKnfIFlEmclpx-9agQywzI3Cpj-I", parse_mode=None)

logfile = str(datetime.date.today()) + '.log' # формируем имя лог-файла

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Send your voice message")

def audio_to_text(dest_name: str):
    r = sr.Recognizer()
    message = sr.AudioFile(dest_name)
    with message as source:
        audio = r.record(source)
    result = r.recognize_google(audio, language="ru_RU")
    return result

def gender_identifier(dest_name: str):
    voice = dest_name
    spect = generate_spectogram(voice)
    img_width, img_height = 432, 288
    female_image = tf.keras.preprocessing.image.load_img(spect, target_size=(img_width, img_height))
    female_img_tensor = tf.keras.preprocessing.image.img_to_array(female_image)
    female_img_tensor = np.expand_dims(female_img_tensor, axis=0)
    female_img_tensor /= 255.
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    result = (loaded_model.predict(female_img_tensor) > 0.5).astype("int32")
    print(result)
    return result



@bot.message_handler(content_types=['voice'])
def get_audio_messages(message):
    try:
        print("Started recognition...")
        file_info = bot.get_file(message.voice.file_id)
        path = os.path.splitext(file_info.file_path)[0]
        fname = os.path.basename(path)
        doc = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(token, file_info.file_path)) #Получаем и сохраняем присланную голосвуху (Ага, админ может в любой момент отключить удаление айдио фай
        with open(fname+'.ogg', 'wb') as f:
            f.write(doc.content) # вот именно тут и сохраняется сама аудио-мессага
        process = subprocess.run(['ffmpeg', '-i', fname+".ogg", fname+'.wav'])
        result = audio_to_text(fname+'.wav')
        result2 = gender_identifier(fname+'.wav')
        if (str(result2) == "[[0]]"):
            file_source, file_destination = fname+'.wav', "reddit_female/"+fname+'.wav'
            shutil.move(file_source, file_destination)
            bot.send_message(message.from_user.id, "Women voice")
        else:
            file_source, file_destination = fname+'.wav', "reddit_male/"+fname+'.wav'
            shutil.move(file_source, file_destination)
            bot.send_message(message.from_user.id, "Men voice")
        bot.send_message(message.from_user.id, "Текст вашего сообщения:\n"+format(result))
    except sr.UnknownValueError as e:
        bot.send_message(message.from_user.id,  "Sorry, I don't understand you or your message is empty...")
        with open(logfile, 'a', encoding='utf-8') as f:
            f.write(str(datetime.datetime.today().strftime("%H:%M:%S")) + ':' + str(message.from_user.id) + ':' + str(message.from_user.first_name) + '_' + str(message.from_user.last_name) + ':' + str(message.from_user.username) +':'+ str(message.from_user.language_code) + ':Message is empty.\n')
    except Exception as e:
        bot.send_message(message.from_user.id,  "Something went wrong...")
        with open(logfile, 'a', encoding='utf-8') as f:
            f.write(str(datetime.datetime.today().strftime("%H:%M:%S")) + ':' + str(message.from_user.id) + ':' + str(message.from_user.first_name) + '_' + str(message.from_user.last_name) + ':' + str(message.from_user.username) +':'+ str(message.from_user.language_code) +':' + str(e) + '\n')
    finally:
        # В любом случае удаляем временные файлы с аудио сообщением
        os.remove(fname+'.ogg')



bot.polling(none_stop=True, interval=0)