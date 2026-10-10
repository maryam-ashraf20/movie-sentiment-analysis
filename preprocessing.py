import re
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text= text.lower()
    #remove html tags
    text= re.sub(r"<[^>]+>"," ",text)
    #remove urls
    text=re.sub(r"https?://\s+|www\.\s+"," ",text)
    #remove extra spaces
    text= re.sub(r"\s+", " ",text)
    #keep letters and spaces only
    text=re.sub(r"[^a-z\s]", '',text).strip()
    return text

lemmatizer= WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith("j"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize_tokens(tokens):
    tagged_token= pos_tag(tokens)

    return [
        lemmatizer.lemmatize(word,get_wordnet_pos(tag))
        for word ,tag in tagged_token
    ]
