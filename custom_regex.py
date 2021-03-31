import re

def get_pattern_count(inputString, pattern):
    r = re.findall(pattern, inputString)
    length = len(r)
    return length

def remove_regex(inputString, pattern):
    r = re.findall(pattern, inputString)
    for i in r:
        inputString = re.sub(i, '', inputString)
    return inputString

def de_emojify(text):
    regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regex_pattern.sub(r'',text)