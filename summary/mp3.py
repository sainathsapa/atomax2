
import speech_recognition as sr
from os import path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import shutil
import moviepy.editor as mp
import subprocess

def create_audio_transcript_file(fileName, lang, fileEXT):
    AUDIO_FILE = ''
    if fileEXT == '.mp4' or fileEXT == '.wmv' or fileEXT == '.avi':
        videofile = path.join(path.dirname(
            path.realpath(__file__)), fileName)
        videoClipFile = mp.VideoFileClip(videofile)

        AUDIO_FILE = path.join(path.dirname(
            path.realpath(__file__)), fileName.replace(fileEXT, '.wav'))
        audioClipFile = videoClipFile.audio.write_audiofile(
            AUDIO_FILE)
    else:
        AUDIO_FILE = path.join(path.dirname(
            path.realpath(__file__)), fileName)

    return get_transcript(AUDIO_FILE, lang)


def get_transcript(path, lang):
    r = sr.Recognizer()

    sound = AudioSegment.from_wav(path)
    chunks = split_on_silence(sound,
                              min_silence_len=100,
                              silence_thresh=sound.dBFS-40,
                              #   silence_thresh=-16,
                              #   keep_silence=1,
                              )

    folder_name = path+"chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened, language=lang)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    shutil.rmtree(folder_name)
    return whole_text


def convert_audio_to_wav(inputFileName):
    inputeFilenameDIR = AUDIO_FILE = path.join(path.dirname(
        path.realpath(__file__)), inputFileName)
    outputeFinaleName = path.join(path.dirname(
        path.realpath(__file__)), inputFileName.replace('.mp3', '.wav'))
    sound = AudioSegment.from_mp3(inputeFilenameDIR)
    sound.export(outputeFinaleName, format="wav")
    return (outputeFinaleName, inputFileName.replace('.mp3', '.wav'))
# fileName = "../files/mp3/test_atom/kishan.wav"
# lang = "te-IN"
# print(create_audio_transcript_file(fileName, lang))
