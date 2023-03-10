from newspaper import Article
import os
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import string
import re
intput_text = ' '
output = ' '


def throughURL(url):

    device = torch.device("cpu")

    # model = ("Bart")
    # Pegasus
    link_paper = (
        url)
    article = Article(link_paper)
    article.download()
    article.parse()
    input_text = article.text
    # print(input_text)
    # st.text(input_text)
    bart_model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-cnn").to(device)
    bart_tokenizer = BartTokenizer.from_pretrained(
        "facebook/bart-large-cnn")
    input_text = ' '.join(input_text.split())

    # print(input_text)

    text = re.sub("https?:\/\/.*[\r\n]*", "", input_text)
    text = re.sub("#", "", text)  # hash nikalre idhar
    # iske lie string library import karna hai
    punct = set(string.punctuation)
    # this is for removing punctuation
    text = "".join([ch for ch in text if ch not in punct])

    # encoding the text to ASCII format
    text = text.encode(encoding="ascii", errors="ignore")

    text = text.decode()  # encoding the text to ASCII format

    # cleaning the text to remove extra whitespace
    clean_text = " ".join([word for word in text.split()])

    clean_text = str(clean_text.lower())

    # print(clean_text)

    input_tokenized = bart_tokenizer.encode(
        clean_text, return_tensors='pt').to(device)

    summary_ids = bart_model.generate(input_tokenized,
                                      num_beams=4, num_return_sequences=1,
                                      no_repeat_ngram_size=2,
                                      length_penalty=1,
                                      min_length=12, max_length=128, early_stopping=True)
    output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
              summary_ids]
    return str(output)
