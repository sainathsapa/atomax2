import yake
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
import language_tool_python
import PyPDF2
import docx
import regex as re
import string
import PyPDF2
import spacy
import torch
from string import punctuation
from numba import jit, cuda
from deepmultilingualpunctuation import PunctuationModel
from newspaper import Article
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import (
    PegasusForConditionalGeneration, PegasusTokenizer, pipeline, BartForConditionalGeneration, BartTokenizer)
from heapq import nlargest
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import speech_recognition as sr
from os import path
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

# ----------------------------------------------------------------
# Translators
# ----------------------------------------------------------------


def translate(summaryText, lang_code):
    translator = Translator()

    Transtxt = translator.translate(summaryText, dest=lang_code)
    return Transtxt


# ----------------------------------------------------------------
#  Convert
# ----------------------------------------------------------------


def filePDF_extract(filePath):

    pdffileobj = open(filePath, 'rb')
    pdfreader = PyPDF2.PdfFileReader(pdffileobj)
    x = pdfreader.numPages
    content = ""
    RetText = ""
    for i in range(x):
        pageobj = pdfreader.getPage(i)
        content = pageobj.extractText()
        RetText = RetText + content
    return RetText


def fileDOCX_extract(filePath):
    doc = docx.Document(filePath)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


def fileTXT_extract(filepath):
    TextFile = open(filepath, "r")
    textRead = TextFile.read()
    TextFile.close()
    return textRead


def URL_extract(url):

    link_paper = (url)
    article = Article(link_paper)
    article.download()
    article.parse()
    input_text = article.text
    input_text = ' '.join(input_text.split())
    return input_text


def setGrammer(inputText):
    model = PunctuationModel()

    inputText = inputText.replace("<n>", "")

    inputText = model.restore_punctuation(inputText)

    my_tool = language_tool_python.LanguageTool('en-US')

    inputText = my_tool.correct(inputText)

    return inputText


def Clean_Text(textforCleanup):
    text = re.sub("https?:\/\/.*[\r\n]*", "", textforCleanup)
    text = re.sub("#", "", text)
    punct = set(string.punctuation)
    text = "".join([ch for ch in text if ch not in punct])
    text = text.encode(encoding="ascii", errors="ignore")
    text = text.decode()
    clean_text = " ".join([word for word in text.split()])
    clean_text = str(clean_text.lower())
    return clean_text


def fromMic(lang_input):
    whole_text = ''
    r = sr.Recognizer()
    breakwordList = ['stop', 'Stop', 'STOP', 'brake', 'Brake', 'BRAKE']

    while (1):

        try:

            with sr.Microphone(device_index=1) as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                MyText = r.recognize_google(audio, language=lang_input)
                MyText = MyText.lower()
                whole_text += MyText+' '
                print(MyText)
                transText = translate(MyText, 'en').text
                print('---------------')
                # print(transText)
                if transText in breakwordList:
                    return whole_text

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown error occurred")


# ----------------------------------------------------------------
# BASIC ALGO
# ----------------------------------------------------------------
device = "cpu"


def PegasusModel(spacy_summary):
    model_name = "google/pegasus-large"

    # Load pretrained tokenizer
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)

    pegasus_model = PegasusForConditionalGeneration.from_pretrained(
        model_name)

    # Create tokens

    tokens = pegasus_tokenizer(
        spacy_summary, truncation=True, padding="max_length", return_tensors="pt")

    encoded_summary = pegasus_model.generate(**tokens)

    # Decode summarized text

    decoded_summary = pegasus_tokenizer.decode(

        encoded_summary[0],

        skip_special_tokens=True

    )

    # print(decoded_summary)

    summarizer = pipeline(

        "summarization",

        model=model_name,

        tokenizer=pegasus_tokenizer,

        framework="pt", truncation=True

    )

    summary = summarizer(spacy_summary, min_length=30, max_length=150)

    summary = summary[0]["summary_text"]

    summary = summary.replace("<n>", "")

    summary = summary.replace("  ", "")

    print(summary)
    return summary


def Spacy(input_text):
    clean_text = str(input_text.lower())
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(clean_text)
    per = 1.0
    tokens = [token.text for token in doc]

    word_frequencies = {}

    for word in doc:

        if word.text.lower() not in list(STOP_WORDS):

            if word.text.lower() not in punctuation:

                if word.text not in word_frequencies.keys():

                    word_frequencies[word.text] = 1

                else:

                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}

    for sent in sentence_tokens:

        for word in sent:

            if word.text.lower() in word_frequencies.keys():

                if sent not in sentence_scores.keys():

                    sentence_scores[sent] = word_frequencies[word.text.lower()]

                else:

                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_tokens) * per)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    # print(summary)
    final_summary = [word.text for word in summary]
    spacy_summary = ' '.join(final_summary)
    return spacy_summary


def Brat(input_text, min_len, mx_len):
    model = "Bart"
    bart_model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-cnn").to(device)
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    input_text = ' '.join(input_text.split())
    input_tokenized = bart_tokenizer.encode(
        input_text, return_tensors='pt').to(device)
    summary_ids = bart_model.generate(input_tokenized,
                                      num_beams=4, num_return_sequences=1,
                                      no_repeat_ngram_size=2,
                                      length_penalty=1,
                                      min_length=min_len, max_length=mx_len, early_stopping=True)
    output = [bart_tokenizer.decode(
        g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


def parse_sentence(input_text, num_return_sequences):
    model_name = 'tuner007/pegasus_paraphrase'
    max_length = 280
    # torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    torch_device = torch.device("cpu")
    model = PegasusForConditionalGeneration.from_pretrained(
        model_name).to(torch_device)
    batch = tokenizer.prepare_seq2seq_batch([input_text], truncation=True, padding='longest', max_length=max_length,
                                            return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, max_length=max_length, num_beams=10, num_return_sequences=num_return_sequences,
                                temperature=1.5)
    a = tokenizer.batch_decode(translated, skip_special_tokens=True)
    print(a)
    return a

   # return tgt_text


# def parse_text_to_new_article(inputeText):
#     splitter = SentenceSplitter(language='en')
#     sentence_list = splitter.split(inputeText)

#     paraphrase = []

#     for i in sentence_list:
#         a = parse_sentence(i, 1)
#         paraphrase.append(a)
#         print("inMAIN - ", paraphrase)

#     paraphrase2 = [' '.join(x) for x in paraphrase]
#     paraphrase3 = [' '.join(x for x in paraphrase2)]
#     paraphrased_text = str(paraphrase3).strip('[]').strip("'")

#     return paraphrased_text

def parse_text_to_new_article(inputeText):
    print("RUN 1 Running")
    output1 = " ".join([my_paraphrase1(sent)
                       for sent in sent_tokenize(inputeText)])
    print("RUN 1 Completed and RUN 2 Starting")
    output2 = " ".join([my_paraphrase1(sent)
                       for sent in sent_tokenize(output1)])
    print("RUN 2 Completed and Parse Sentence Running")
    op = parse_sentence(output2, 500)
    return op


tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


def my_paraphrase1(sentence):
    sentence = "paraphrase:" + sentence+"</s>"
    encoding = tokenizer.encode_plus(
        sentence, padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=256,
                             do_sample=True, top_k=120, top_p=0.95, early_stopping=True, num_return_sequences=1)
    output1 = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output1


def genMeta(text):
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 20
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    retList = []
    for k in keywords:
        retList.append(k[0])
    return retList


def GenArticleHTML(text):
    html = "<html><head></head><body>"+text+"</body></html>"
    soup = BeautifulSoup(html)

    print("BRAT RUNNING")
    # desc = Brat(text, 50, 90)
    # desc = Brat(text, 50, 90)
    desc = "text"
    print(desc)

    print("BRAT RUNNING")
    # title = Brat(desc, 10, 30)
    title = "TestTitle"
    print(title)
    metaDict = {
        'content': 'text/html',
        'description': desc,
        'robots': 'index, archive',
        'twitter:card': desc,
        'twitter:image:src': 'https://pbs.twimg.com/ext_tw_video_thumb/560070131976392705/pu/img/TcG_ep5t-iqdLV5R.jpg',
        'og:title': title,
        'keywords': ','.join(genMeta(text))

    }
    # metaTags.attrs[''] =
    # metaTags.attrs[''] = ''
    # metaTags.attrs[''] =
    # metaTags.attrs[''] =
    # metaTags.attrs[''] =
    # separator = ', '
    metaTags = soup.new_tag('meta')

    for i, j in metaDict.items():
        metaTags = soup.new_tag('meta')

        metaTags.attrs['name'] = i
        metaTags.attrs['content'] = j

        # tag = soup.(metaTags)

        soup.head.append(metaTags)
    # metaTags.attrs[''] =
    properties = {
        'article:publisher': 'https://www.facebook.com/HBR',
        'article:modified_time': '2021-08-30T16:17:26Z',
        'og:image': 'https://hbr.orghttps://hbr.org/resources/images/article_assets/2020/03/Mar20_06_1174180788.jpg',
        'article:published_time': '2020-03-06T13:25:51Z',
        'og:url': 'https://hbr.org/2020/03/the-key-to-inclusive-leadership',
        'article:tag': ''.join(genMeta(text)),
        'og:site_name': 'Harvard Business Review'
    }

    for i, j in properties.items():
        metaTags = soup.new_tag('meta')
        metaTags.attrs['property'] = i

        metaTags.attrs['content'] = j

        # tag = soup.(metaTags)

        soup.head.append(metaTags)
    ret = {}
    ret['text'] = text
    ret['meta'] = soup.prettify()
    return ret

# print(PegasusModel(Clean_Text(URL_extract(c
#     ("https://www.indiatoday.in/science/story/russia-kamchatka-peninsula-volcanoes-awaken-major-eruption-warning-ash-lava-2299836-2022-11-21")))))
