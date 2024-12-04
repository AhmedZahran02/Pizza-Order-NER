from var import *
import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

class Normalizer:
    ''' 
        Removes All Punctuation From The Given String 
        Punctuation will is defined in var.py
    '''
    def remove_punctuations(self, TOP):
        regex = "".join([ fr"\{punc}" for punc in PUNCTUATIONS ])
        TOP = re.sub(fr"[{regex}]", '', TOP)
        return TOP
    
    '''
        Removes Words From The Given String Which Are Defined In var.py
        Note: This Words Is Assumed To Be Stemmed Already
    '''
    def remove_words(self, TOP):
        regex = fr"(?<=(?:\b|^))(?:{"|".join(BLACKLIST)})(?=(?:\b|&))"
        return re.sub(fr"({regex})", '', TOP)

    def replace_numbers(self, TOP):
        regex = r'\b(?:\d+|a)\b' # TODO: revise this
        TOP = re.sub(regex, 'DIGIT', TOP)
        return TOP

    '''
        Follow Rules Of Replacing Some Words With Other In Case Of Needing
    '''
    def reconstruct_words(self, TOP):
        return TOP
    
    '''
        - Replace Multiple Spaces With Only One Space
        - Removes Spaces At The Beginning And End Of The Given String
    '''
    def reorganize_spaces(self, TOP):
        TOP = re.sub(r'\s+', ' ', TOP)
        return re.sub(r'(?:\s+$)|(?:^\s+)', '', TOP)
    
    '''
        Stems the given word and convert it to lowercase
    '''
    def stem_word(self, token):
        return stemmer.stem(token, to_lowercase=False)

    '''
        Stems the given sentence and convert all words to lowercase
    '''
    def stem_sentence(self, sentence):
        return " ".join([ self.stem_word(word) for word in sentence.split(' ') ])

    '''
        Normalizes the given sentence by performing the following steps:
        1. Reconstructs the words using the rules defined in var
        2. Removes punctuation defined in var
        3. Reorganizes spaces
        4. Stems the sentence
        5. Returns the normalized sentence
    '''
    def normalize(self, sentence):
        NEXT = self.remove_punctuations(sentence) 
        NEXT = self.remove_words(NEXT)
        NEXT = self.replace_numbers(NEXT)
        NEXT = self.reconstruct_words(NEXT)
        NEXT = self.reorganize_spaces(NEXT)
        NEXT = self.stem_sentence(NEXT)
        return NEXT
    
