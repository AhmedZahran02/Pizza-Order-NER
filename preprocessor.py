from libraries import *
import var

class Normalizer:
    ''' 
        Removes All Punctuation From The Given String 
        Punctuation will is defined in var.py
    '''
    def remove_punctuations(self, TOP):
        regex = "".join([ fr"\{punc}" for punc in PUNCTUATIONS ])
        TOP = re.sub(fr"([{regex}])", r' \1 ', TOP)
        return TOP
    
    '''
        Removes Words From The Given String Which Are Defined In var.py
        Note: This Words Is Assumed To Be Stemmed Already
    '''
    def remove_words(self, TOP):
        regex = fr"(?<=(?:\b|^))(?:{"|".join(BLACKLIST)})(?=(?:\b|&))"
        return re.sub(fr"({regex})", '', TOP)

    def replace_numbers(self, TOP):
        regex = r'\b([1-9]|1[0-9]|20)\b'
        TOP = re.sub(regex, 'SM_NUM', TOP)

        regex = r'\b([2-9][1-9]|[3-9][0-9]|[1-9][0-9]{2,})\b'
        TOP = re.sub(regex, 'LG_NUM', TOP)

        regex = fr"(?<=(?:\b|^))(?:{"|".join(EN_NUMS)})(?=(?:\b|&))"
        TOP = re.sub(regex, "SM_NUM", TOP)

        return TOP
    
    def replace_numbers_and_keep(self, TOP):
        ANS = []
        for word in TOP.split(" "):
            NEXT = self.replace_numbers(word)
            if NEXT.endswith("NUM"):
                ANS.append(f"{NEXT}_{word}")
            else:
                ANS.append(f"{word}")
        return " ".join(ANS)

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
        NEXT = sentence.replace("COMPLEX_TOPPING", "COMPLEXTOPPING")
        #
        NEXT = self.remove_punctuations(NEXT) 
        # NEXT = self.remove_words(NEXT)
        NEXT = self.replace_numbers(NEXT)
        # NEXT = self.reconstruct_words(NEXT)
        NEXT = self.reorganize_spaces(NEXT)
        # NEXT = self.stem_sentence(NEXT)
        return NEXT
    

# norm = Normalizer()

# x = norm.normalize("twenty pizza with green-onions, 15 with ice, 500 ml cola")

# print(x)