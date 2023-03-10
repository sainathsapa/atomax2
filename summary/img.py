import subprocess
import os.path
from subprocess import STDOUT, PIPE
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
import datetime


fileDIR = "files/"


def text_to_image(text, userName):
    font = ImageFont.truetype("C:\\Users\\djsai\\Downloads\\Vani.ttf", 40)
    para = textwrap.wrap(text, width=100)
    finall = '\n'.join(para)
    text = finall
    image_width = 2250
    im = Image.new("RGB", size=(2480, 1748),  color=(255, 255, 255))
    d = ImageDraw.Draw(im)
    lines = len(para)
    width = 2480
    y = int(width/2)-(70*(lines-1))
    # y= 10
    x = 250
    for line in text.split("\n"):
        words = line.split(" ")

        words_length = sum(d.textlength(w, font=font) for w in words)
        space_length = (image_width - words_length) / (len(words) - 1)
        x = 100
        for word in words:
            d.text((x, y), word, font=font, fill="black")
            x += d.textlength(word, font=font) + space_length
        y += 50
    # margin
    # top
    d.line((50 - 15, 50) + (2450, 50), fill=0)
    # left
    d.line((50 - 15, 50) + (50 - 15, 1700), fill=0)
    # bottom
    d.line((50 - 15, 1700) + (2450, 1700), fill=0)
    # right
    d.line((2450, 50) + (2450, 1700), fill=0)

    ts = datetime.datetime.now()

    ts = str(int(ts.strftime("%Y%m%d%H%M%S")))

    imgFileName = os.path.join(fileDIR, userName)+"/" + \
        userName+"_"+ts+"_"+".png"
    im.save(imgFileName)
    return imgFileName


# def compile_java(java_file):
#     subprocess.check_call(['javac', java_file])


# def execute_java(java_file, stdin):
#     java_class, ext = os.path.splitext(java_file)
#     cmd = ['java', java_class]
#     proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
#     stdout, stderr = proc.communicate(stdin)
#     print('This was "' + stdout + '"')


# def run_img(text, usrname):
#     compile_java('TextToGraphics.java')
#     execute_java('TextToGraphics', 'Jon')


# print(run_img('text', 1))