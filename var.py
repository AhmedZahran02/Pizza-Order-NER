from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

################################################################
################### PREPROCESSOR PARAMETERS ####################
################################################################

PUNCTUATIONS=[
    ".",  # Period
    ",",  # Comma
    ";",  # Semicolon
    ":",  # Colon
    "!",  # Exclamation Mark
    "?",  # Question Mark
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

NEGATING_VERBS = [
    "avoid",
    "deny",
    "refuse",
    "prevent",
    "reject",
    "stop",
    "prohibit",
    "resist",
    "escape",
    "forbid",
    "hinder",
    "discourage",
    "preclude",
    "neglect",
    "escape",
    "block",
    "suppress",
    "oppose",
    "shun",
    "ignore",
    "eliminate",
    "exclude",
    "abstain",
    "detest",
    "proscribe",
    "terminate",
    "obstruct",
    "decline",
    "abandon",
    "disapprove",
    "withhold",
    "counteract",
    "nullify",
    "refrain"
]

NEGATING_IN = [
    "without", "against", "despite", "except", "beyond", "excluding"
]

SMALL_NUMBERS = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
]

PIZZA_WORDS = ["pizza", "pie", "slice"]

BLACKLIST    = list(set([ stemmer.stem(x) for x in [] ]))
PIZZA_WORDS  = list(set([ stemmer.stem(x) for x in PIZZA_WORDS]))

################################################################
###################### MODEL PARAMETERS ########################
################################################################

EMBEDDING_SIZE = 100
HIDDEN_SIZE = 128
