from var import *
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import download
from nltk.corpus import wordnet
import spacy
nlp = spacy.load("en_core_web_sm")


download('wordnet')
download('omw-1.4')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN 
    elif tag.startswith('R'):
        return wordnet.ADV 
    else:
        return wordnet.NOUN


def POS_rules(word, tag):
    original_tag = tag[ tag.find("_") + 1 : ]    
    if original_tag in [ "CC", "CD", "VB", "VBD", "VBG", "VBN", "IN" ] :
        return tag
    elif original_tag in [ "EX", "LS", "MD", "PRP", "PRP$", "RBR", "RBS", "SYM", "TO", "UH" ]:
        return None
    else:
        return lemmatizer.lemmatize( word, get_wordnet_pos(original_tag) )

class Normalizer:
    ''' 
        Removes All Punctuation From The Given String 
        Punctuation will is defined in var.py
    '''
    def remove_punctuations(self, TOP):
        regex = "".join([ fr"\{punc}" for punc in PUNCTUATIONS ])
        TOP = re.sub(fr"[{regex}]", ' ', TOP)
        return TOP
    
    '''
        Removes Words From The Given String Which Are Defined In var.py
        Note: This Words Is Assumed To Be Stemmed Already
    '''
    def remove_words(self, TOP):
        NEXT = []
        for word, tag in TOP:
            replacement = POS_rules(word, tag)
            if replacement is not None: NEXT.append((word, tag))
        return NEXT

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
        NEXT = ""
        for word, tag in sentence:
            NEXT += self.stem_word(word) + " "
            
        return " ".join([ self.stem_word(word) for word in sentence.split(' ') ])
    
    def lemmatize_sentence(self, TOP):
        NEXT = []
        for word, tag in TOP:
            NEXT.append((self.lemmatize_word(word, get_wordnet_pos(tag)), tag))
        return NEXT
    
    def lemmatize_word(self, word, tag):
        return lemmatizer.lemmatize(word, tag)

    def attach_pos_tags(self, sentence):
        doc = nlp(sentence)
        NEXT = []
        for token in doc:
            if token.tag_ == "IN" and token.text in NEGATING_IN:
                NEXT.append((token.text, "NEG_" + token.tag_))
            elif get_wordnet_pos(token.tag_) == wordnet.VERB and token.text in NEGATING_VERBS:
                NEXT.append((token.text, "NEG_" + token.tag_))
            elif token.tag_ == "CD" and token.text not in SMALL_NUMBERS:
                NEXT.append((token.text, "LG_CD"))
            else:
                NEXT.append((token.text, token.tag_))
        return NEXT

    '''
        Normalizes the given sentence by performing the following steps:
        1. Reconstructs the words using the rules defined in var
        2. Removes punctuation defined in var
        3. Reorganizes spaces
        4. Stems the sentence
        5. Returns the normalized sentence
    '''
    def normalize(self, sentence, lemmatize=True):
        NEXT = sentence.lower()
        NEXT = self.reconstruct_words(NEXT)
        NEXT = self.reorganize_spaces(NEXT)
        NEXT = self.remove_punctuations(NEXT) 
        NEXT = self.attach_pos_tags(NEXT)
        NEXT = self.remove_words(NEXT)
        return self.lemmatize_sentence(NEXT) if lemmatize else NEXT
    
# norm = Normalizer()
# normalized = norm.normalize("i want two large banana pepperoni pizzas avoid any onions and no potato")
# print(normalized)