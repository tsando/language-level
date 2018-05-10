import pandas as pd
import numpy as np
import datetime
import time
import re
import pickle as pickle

import nltk
from nltk.corpus import wordnet

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV  # finding model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


##############################################################
#               FEATURE ENGINEERING
##############################################################


class CTextFeatures:

    def __init__(self, text):
        self.text = text
        self.nchars = len(text)

        self.tokenizedd = False
        self.tokens = None
        self.ntokens = None
        self.tokens_len = None
        self.sent_tokens = None
        self.nsent = None
        self.sen_len = None

        self.features = None
        self.pos_tags = None
        self.lemmas = None
        self.level_inc = None
        self.level_counts = None

    def tokenize(self):
        self.tokenizedd = True
        self.tokens = nltk.word_tokenize(self.text)  # already normalised if used self.clean_text beforehand
        self.ntokens = len(self.tokens)
        self.tokens_len = [len(x) for x in self.tokens]  # use isalpha to ignore punctuation?
        self.sent_tokens = nltk.sent_tokenize(self.text)
        self.nsent = len(self.sent_tokens)
        self.sen_len = [len(x) for x in self.sent_tokens]

    def get_features(self, lex_db=None, lmtzr=None):

        if not self.tokenizedd:
            self.tokenize()

        self.features = {
            # Length-based
            'nchars': self.nchars,
            'ntokens': self.ntokens,
            'nsent': self.nsent,
            'lexical_diversity': len(set(self.tokens)) / self.ntokens,  # number of distinct words wrt the total number of words
            'avg_sentence_length': np.mean(self.sen_len),
            # 'sentence_length_by_tokens': np.mean(self.sen_len) / self.ntokens,
            # 'sentence_length_by_chars': np.mean(self.sen_len) / self.nchars,
            'avg_token_length': np.mean(self.tokens_len),
            'nlongwords': len([x for x in self.tokens_len if x > 10]),
            'avg_tokens_per_sent': self.ntokens / self.nsent,  # could be biased by stop-words?
        }
        # Lexical
        if lex_db and lmtzr:
            # join into single dict
            self.features = {**self.features, **self.get_lexical_features(lex_db, lmtzr)}
        return self.features

    def get_lexical_features(self, lex_db, lmtzr):

        if not self.tokenizedd:
            self.tokenize()

        # See https://stackoverflow.com/questions/771918/how-do-i-do-word-stemming-or-lemmatization
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        # We need POS Tagging before word lemmatization if not already done to get grammar features
        if self.pos_tags is None:
            self.pos_tags = nltk.pos_tag(self.tokens)  # note must use tokens not types to detect correct POS
        # ensure no punctuation included
        self.lemmas = [(lmtzr(x[0], get_wordnet_pos(x[1])), x[1]) for x in self.pos_tags if x[0].isalpha()]

        self.level_counts = {'A1': 0,
                             'A2': 0,
                             'B1': 0,
                             'B2': 0,
                             'C1': 0,
                             'C2': 0,
                             'Unknown': 0
                             }

        mappings = {
            'DT': 'determiner',
            'PDT': 'determiner',
            'WDT': 'determiner',
            'VB': 'verb',
            'VBD': 'verb',
            'VBG': 'verb',
            'VBN': 'verb',
            'VBP': 'verb',
            'VBZ': 'verb',
            'IN': 'preposition',
            'TO': 'preposition',
            'CC': 'conjunction',
            'PR': 'pronoun',
            'PRP': 'pronoun',
            'PRP$': 'pronoun',
            'WP': 'pronoun',
            'WP&': 'pronoun',
            'JJ': 'adjective',
            'JJR': 'adjective',
            'JJS': 'adjective',
            'RB': 'adverb',
            'RBR': 'adverb',
            'RBS': 'adverb',
            'WRB': 'adverb',
            'MD': 'modal verb',
            'CD': 'number',
            'NN': 'noun',
            'NNP': 'noun',
            'NNPS': 'noun',
            'NNS': 'noun'
        }

        for t in self.lemmas:
            word = t[0]
            pos_tag = t[1]
            try:
                d = lex_db[word]
                if len(d) > 1:
                    # Only include first two chars of pos_tag
                    level = d[mappings[pos_tag]]
                else:
                    level = list(d.values())[0]
                # print(word, level, pos_tag)
                self.level_counts[level] += 1
            except KeyError:
                self.level_counts['Unknown'] += 1
                # print('{} failed!!!! ---> {}'.format(word, pos_tag))
                pass
        # Normalise to incidence scores (per total tokens)
        self.level_inc = {k + '_inc': v / self.ntokens for k, v in self.level_counts.items()}
        return self.level_inc

    def clean_text(self):
        # replace very spefific email format found in one entry.
        self.text = re.sub('From:[A-Za-z0-9]+@[A-Za-z0-9]+.com ', ' ', self.text)
        self.text = re.sub('To:[A-Za-z0-9 ]+@[A-Za-z0-9]+.com ', ' ', self.text)
        self.text = re.sub('Date:[A-Za-z0-9 ,]+:[0-9]+[AP]M ', ' ', self.text)

        # normalise punctuation to just .
        punctuation_dic = {",": " ",  ### REMOVES ALL COMMAS
                           ";": " ",  ### REMOVES ALL Semi colons
                           ":": " ",  ### removes semicolons
                           "!": ".",  ### converts ! to .
                           "?": ".",  ### converts ? to .
                           }
        for entry in punctuation_dic:
            self.text = self.text.replace(entry, punctuation_dic[entry])

        # remove multiple .
        self.text = re.sub('\.+', '.', self.text)

        # add spaces after punctuation, remove repeated spaces, remove spaces before punctuation.
        self.text = self.text.replace('.', '. ')
        self.text = re.sub(' +', ' ', self.text)
        self.text = self.text.replace(' .', '.')

        # reduce hidden information '#' to just one symbol
        self.text = self.text.replace('# #', '#')
        self.text = re.sub('#+', '#', self.text)

        # replace quotation character with proper symbol in text
        self.text = self.text.replace('&quot;', "'")

        # replace newline character
        self.text = self.text.replace('\n', '')

        # remove leading and trailing white space of entire text block
        self.text = self.text.lstrip().rstrip()

        # ### normalise to lowercase ?????????????????????????
        self.text = self.text.lower()

        return self.text


def get_lex_db():
    temp = pd.read_csv('data/lexical_en_m3.csv')
    # clean-up
    temp['Part of Speech.1'] = temp['Part of Speech.1'].apply(lambda x: x.replace('“', '').replace('”', ''))
    temp['Part of Speech'] = temp['Part of Speech'].apply(lambda x: str(x).lower())
    # Get into dict
    temp = temp[['Word', 'Part of Speech', 'Part of Speech.1']].to_dict('index')

    lex_db = {}
    for k1, v1 in temp.items():
        if v1['Word'] not in lex_db:
            lex_db[v1['Word']] = {v1['Part of Speech']: v1['Part of Speech.1']}
        else:
            lex_db[v1['Word']][v1['Part of Speech']] = v1['Part of Speech.1']

    # lex_db = {v1['Word']: v1['Part of Speech.1'] for k1, v1 in lex_db.items()}

    return lex_db


##############################################################
#               PIPELINES
##############################################################


def build_features(df):
    # Get text features
    start_time = time.time()

    print('Cleaning text...')
    df['text_cleaned'] = df['text'].apply(lambda x: pd.Series(CTextFeatures(x).clean_text()))

    print('Building features...')
    lmtzr = nltk.WordNetLemmatizer().lemmatize
    lex_db = get_lex_db()
    df = df.join(df['text_cleaned'].apply(lambda x: pd.Series(CTextFeatures(x).get_features(lex_db, lmtzr))))
    names = list((CTextFeatures(df.loc[0, 'text_cleaned']).get_features(lex_db, lmtzr)).keys())
    print_runtime(start_time)
    return df, names


def run_simple(X, y, model):
    # split the data with 50% in each set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

    # fit the model on one set of data
    model.fit(X_train, y_train)

    # evaluate the model on the second set of data
    y_test_model = model.predict(X_test)
    print(accuracy_score(y_test, y_test_model))


##############################################################
#               HELPER FUNCTIONS
##############################################################


def get_csv_from_xml():
    """
    Function to transform xml data to csv so faster processing.
    WARNING: takes about 30 mins to run!
    NOTES:
        - manually removed the header/meta data section of the xml and saved it as EFWritingData_nohead_replaced.xml
        - renamed the <text> to <userinput> using following command:
            sed -i '' -e 's/text>/userinput>/g' EFWritingData_nohead_replaced.xml
    :return: pandas df saved as csv
    """

    from bs4 import BeautifulSoup

    with open("EFWritingData_nohead_replaced.xml") as fp:
        soup = BeautifulSoup(fp, 'xml')

    ll = soup.find_all('writing')
    entries = []

    for w in ll:
        entries.append(
            {'writing_id': w['id'],
             'writing_level': w['level'],
             'writing_unit': w['unit'],
             'topic_id': w.topic['id'],
             'topic_text': w.topic.get_text(),
             'date': w.date.get_text(),
             'grade': w.grade.get_text(),
             'text': w.userinput.get_text()})

    df = pd.DataFrame(entries)
    df.to_csv('EFWritingData_parsed.csv')

    return df


def print_runtime(start_time):
    end_time = time.time()
    run_time = end_time - start_time
    print('Run time: {}h:{}m:{}s'.format(int(run_time / 3600), int(run_time / 60) % 60, int(run_time % 60)))


def save_pickle(object, path_and_name):
    # Saves pickle
    with open(path_and_name, 'wb') as fp:
        pickle.dump(object, fp)
    pass


def open_pickle(path_and_name):
    # Opens pickle - note required 'rb' inside open given above 'wb'
    with open(path_and_name, 'rb') as fp:
        object = pickle.load(fp)
    return object


def prepare_and_save_pickle(size=100000, name='df.p', clean=False):
    print('In prepare_and_save_pickle... took:')
    start_time = time.time()

    df = pd.read_csv('data/EFWritingData_parsed.csv')

    # Rename index column
    df = df.rename(columns={'Unnamed: 0': 'idx'})

    # Slim data to make it faster
    df = df.iloc[:size]

    if clean:
        #  Convert string date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Clean text
        df['text_cleaned'] = df['text'].apply(lambda x: CTextFeatures(x).clean_text())

    save_pickle(df, 'data/' + name)

    print_runtime(start_time)

    pass


##############################################################
#                   MAIN
##############################################################


if __name__ == "__main__":

    # ############      Single text example     ############
    # text = "\n      After some time, the affection between them is progressing well. " \
    #        "John's personality deeply moved Isabella. So Isabella decided to break up with Tom " \
    #        "and fell in love with John. John also feeled that Isabella was the woman he loved deeply. " \
    #        "To his joy, he could find his true love during his travel. In the end, they married together.\n    "
    #
    # lmtzr = nltk.WordNetLemmatizer().lemmatize
    # lex_db = get_lex_db()
    #
    # tf = CTextFeatures(text)
    # tf.clean_text()
    # tf.get_features(lex_db, lmtzr)

    # ############      10k samples example     ############

    # prepare_and_save_pickle(10000, '10k.p')
    df = open_pickle('data/10k.p')

    df, derived_features = build_features(df)

    # Select only these to benchmark vs v2
    derived_features = ['ntokens',
                        'nsent',
                        'avg_tokens_per_sent',
                        'lexical_diversity',
                        'avg_token_length',
                        'nlongwords',
                        'nchars',
                        'A1_inc',
                        'A2_inc',
                        'B1_inc',
                        'B2_inc',
                        'C1_inc',
                        'C2_inc']

    X = df[derived_features]
    y = df['writing_level']
    model = SVC()

    run_simple(X, y, model)
    pass
