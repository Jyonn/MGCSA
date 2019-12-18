from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
import json


def bleu(gts, res):
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    print('belu = %s' % score)


def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)


def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)


def main():
    epoch = 3
    dataset = 'TACoS'
    with open('examples/%sepoch%sgts.json' % (dataset, epoch), 'r') as file:
        gts = json.load(file)
    with open('examples/%sepoch%sres.json' % (dataset, epoch), 'r') as file:
        res = json.load(file)

    bleu(gts, res)
    meteor(gts, res)
    rouge(gts, res)


main()
