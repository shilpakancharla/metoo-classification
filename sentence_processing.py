import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

l = nltk.stem.WordNetLemmatizer()

def parts_of_speech_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_lemma(sentence):
    # Find all parts of speech for each token
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    word_net_tagged = map(lambda x: (x[0], parts_of_speech_tag(x[1])), tagged)
    lemmatized_sentence = []
    for word, tag in word_net_tagged:
        if word == 'ass':
            lemmatized_sentence.append(word)
        elif tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(l.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)