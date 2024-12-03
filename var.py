from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


PUNCTUATIONS=[
    ".",  # Period
    ",",  # Comma
    ";",  # Semicolon
    ":",  # Colon
    "!",  # Exclamation Mark
    "?",  # Question Mark
    "'",  # Apostrophe
    '"',  # Quotation Marks
    "_",  # Em Dash
    "-",  # Hyphen
    "[",  # Left Bracket
    "]",  # Right Bracket
    "{",  # Left Brace
    "}",  # Right Brace
    "/",  # Slash
    "\\", # Backslash
    "|",  # Vertical Bar
    "@",  # At Symbol
    "#",  # Hash
    "$",  # Dollar Sign
    "%",  # Percent
    "^",  # Caret
    "&",  # Ampersand
    "*",  # Asterisk
    "_",  # Underscore
    "+",  # Plus
    "=",  # Equals
    "<",  # Less Than
    ">",  # Greater Than
    "~",  # Tilde
    "`"   # Grave Accent
]

BLACKLIST = [
    # human references
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yourself',
    'he', 'him', 'hi', 'himself',
    'she', 'her', 'herself',
    'it', 'itself',
    'we', 'us', 'our', 'ourselv',
    'they', 'them', 'their', 'themselv',
    'person', 'peopl', 'human',
    'individu', 'man', 'men',
    'woman', 'women', 'child', 'children',
    'adult', 'someon', 'somebodi',
    'anyon', 'anybodi', 'everyon', 'everybodi',
    'no on', 'nobodi'

    # admiring something (who cares cry about it)
    "love", "like", "admire", "adore", "cherish", 
    "appreciate", "respect", "idolize", "enjoy", 
    "value", "revere", "treasure", "favor", 
    "prefer", "esteem", "venerate", "worship", 
    "fancy", "savor", "delight", "care"

    # extra
    'the', 'and', 'or', 'with', 'but', 'within', 'to', 'by', 

    'id', 'ive', 'iam', 'along', 'dont', 'doesnt', 'on', 'in', 'over'
]

PIZZA_WORDS = ["pizza", "pie", "slice"]



BLACKLIST    = list(set([ stemmer.stem(x) for x in BLACKLIST ]))
PIZZA_WORDS  = list(set([ stemmer.stem(x) for x in PIZZA_WORDS]))
