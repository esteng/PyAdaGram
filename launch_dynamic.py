import os
import re
import sys
from collections import defaultdict
import numpy.random
import cPickle
import string
import numpy
import getopt
import sys
import random
import time
import math
import re
import pprint
import codecs
import datetime
import optparse
import os;
import nltk;
import numpy;
import scipy;
import scipy.io;
import collections;
from nltk.grammar import read_grammar, standard_nonterm_parser, Production, Nonterminal
import hybrid;

def generate_sentence():
    length = numpy.random.randint(3,12,1)[0]
    values = ["".join([str(x) for x in numpy.random.randint(0,9,3)]) for i in range(length)]

    return " ".join(values)

def generate_10():
    used = set()
    sents = []
    for i in range(10):
        sent = generate_sentence()
        used |= set(sent.split(" "))
        sents.append(sent)


    return sents, used







if __name__ == '__main__':
    desired_truncation_level = {};
    alpha_pi = {};
    beta_pi = {};
    adapted_non_terminals = set();
    adapted_non_terminal = nltk.Nonterminal("Word");
    adapted_non_terminals.add(adapted_non_terminal);
    desired_truncation_level[adapted_non_terminal] = 1500
    alpha_pi[adapted_non_terminal] = float(100);
    beta_pi[adapted_non_terminal] = float(0);
    snapshot_interval = 10
    output_directory = "/Users/elias/PyAdaGram/dynamic"

    grammar_rules = """Words -> Word
                        Words -> Word Words
                        Word -> Chars
                        Chars -> Char
                        Chars -> Char Chars
                        Char -> "100"
                        Char -> "200"
                         """

    start, productions = read_grammar(grammar_rules, standard_nonterm_parser, probabilistic=False)


    adagram_inferencer = hybrid.Hybrid(start,
                                       productions,
                                       adapted_non_terminals
                                       );

    adagram_inferencer._initialize(1000,
                                   10,
                                   1.0,
                                   0.5,
                                   alpha_pi,
                                   beta_pi,
                                   None,
                                   desired_truncation_level,
                                   10
                                   );

    i = 0
    training_clock = time.time()
    training_iterations = 1000
    while i < 100:
        ten_sents, used = generate_10()
        adagram_inferencer._terminals |= used
        production_list = [Production(Nonterminal("Char"), [terminal]) for terminal in used]
        adagram_inferencer.update_grammar(production_list)
        # for iteration in range(training_iterations):
        # pick up again here!
        clock_iteration = time.time();
        number_of_processes = 1;
        clock_e_step, clock_m_step = adagram_inferencer.learning(ten_sents, number_of_processes);

        if (i+1)%snapshot_interval==0:
            adagram_inferencer.export_adaptor_grammar(os.path.join(output_directory, "adagram-" + str((i+1))))

        if (i+1) % 1000==0:
            snapshot_clock = time.time() - snapshot_clock;
            print 'Processing 1000 mini-batches take %g seconds...' % (snapshot_clock);
            snapshot_clock = time.time()
        clock_iteration = time.time()-clock_iteration;
        print 'E-step, M-step and iteration %d take %g, %g and %g seconds respectively...' % (adagram_inferencer._counter, clock_e_step, clock_m_step, clock_iteration);
    
        i+=1    
    adagram_inferencer.export_adaptor_grammar(os.path.join(output_directory, "adagram-" + str(adagram_inferencer._counter+1)))
    cpickle_file = open(os.path.join(output_directory, "model-%d" % (i+1)), 'wb');
    cPickle.dump(adagram_inferencer, cpickle_file);
    cpickle_file.close();
    
    training_clock = time.time()-training_clock;
    print 'Training finished in %g seconds...' % (training_clock);


