import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)

def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary = {}

    for root, files in os.walk(directory):
        for file in files:
            x = open(os.path.join(root, file), "r")
            dictionary[file] = x.read()


    return dictionary


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    punct = string.punctuation
    stop = nltk.corpus.stopwords.words("english")

    words = nltk.word_tokenize(document.lower())
    words = [word for word in words if word not in punct and word not in stop]

    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    IDF = dict()
    total_num = len(documents)
    words = set(sum(documents.values(), []))
    
    for word in words:
        count = 0

        for doc in documents.values():
            if word in doc:
                count += 1

        idf = math.log(total_num/count )

        IDF[word] = idf 

    return IDF


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    score = dict()

    for file, words in files.items():
        count = 0
        for word in query:
            count += words.count(word) * idfs[word]
        score[file] = count

    ordered_score = sorted(score.items(), key=lambda x: x[1], reverse = True)
    ordered_score = [x[0] for x in ordered_score]

    return ordered_score[:n] 

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores = {}

    for sentence, sent in sentences.items():
        score = 0
        for word in query:
            if word in sent:
                score += idfs[word]

        if score != 0:
            dens = sum([sent.count(x) for x in query]) / len(sent)
            scores[sentence] = (score, dens)

    sorted_score = [a for a, b in sorted(scores.items(), ley = lambda x: x(x[1][0], x[1][1]),
    reverse = True)]

    return sorted_score

if __name__ == "__main__":
    main()
