import os
import numpy as np
from scipy.spatial.distance import cdist

from web.embedding import Embedding

def fetch_toefl():
    parent = os.path.dirname(os.path.abspath(__file__))
    QUESTIONS_PATH = os.path.join(parent,'../../synonym_data/toefl/toefl.qst')
    ANSWERS_PATH = os.path.join(parent,'../../synonym_data/toefl/toefl.ans')
    print(QUESTIONS_PATH)

    def read_questions():
        with open(QUESTIONS_PATH, 'r') as q:
            questions = []
            lines = list(q.readlines())
            cursor = 0
            complete = False
            while not complete:
                line = lines[cursor].strip()
                cursor += 1
                if line:
                    _, question = line.split('\t')
                    answers = [lines[cursor + i].strip().split('\t')[1] for i in range(4)]
                    questions.append((question, answers))
                    cursor += 4
                complete = len(questions) == 80
        return questions

    def read_answers():
        with open(ANSWERS_PATH, 'r') as a:
            answers = []
            for line in a.readlines():
                line = line.strip()
                if line:
                    answer = line.split('\t')[-1]
                    assert answer in 'abcd', 'Invalid answer'
                    order = ord(answer) - ord('a')
                    answers.append(order)
        assert len(answers) == 80, 'Wrong number of lines in ' + ANSWERS_PATH
        return answers

    questions = read_questions()
    answers = read_answers()
    toefl = [(q,options,a) for ((q,options),a) in zip(questions,answers)]
    return toefl

def evaluate_synonyms(e, problems):

    correct = 0
    total = 0

    #debugging...
    if not e:
        all_words = np.concatenate([[q]+o for q,o,_ in problems])
        e = Embedding.from_dict({w: np.random.random(10) for w in all_words})

    meanvec = np.mean(e.vectors, axis=0)

    # with open('synonyms_test_words', 'a') as testw:
    for question,options,answer in problems:
        # testw.write('\n'.join(options+[question])+'\n')
        if question in e:
            print('question: ' + question)
            print(options)
            q_v = e[question].reshape(1, -1)
            q_ops = np.vstack([e[op] if op in e else meanvec for op in options])
            distances = cdist(q_v, q_ops, metric='cosine')[0]
            selected = np.argsort(distances)[0]
            if selected==answer:
                correct+=1
        total += 1

    score = correct*1./total

    return score

def evaluate_TOEFL(e):
    return evaluate_synonyms(e, fetch_toefl())

def evaluate_ESL(e):
    return evaluate_synonyms(e, fetch_esl())

def fetch_esl():
    parent = os.path.dirname(os.path.abspath(__file__))
    ESL_PATH = os.path.join(parent, '../../synonym_data/esl/esl.txt')
    esl = []
    with open(ESL_PATH, 'r') as fin:
        for line in fin:
            if line.startswith('#') or not line.strip(): continue
            words = line.strip().split(' | ')
            assert len(words) == 5, 'Ill-formed line '+line
            question=words[0]
            options=words[1:]
            answer=0
            esl.append((question,options,answer))
    return esl


if __name__ == '__main__':
    score = evaluate_TOEFL(None)
    print(score)
    score = evaluate_ESL(None)
    print(score)


