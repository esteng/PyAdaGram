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
import itertools
from util import NoisyProduction


def substitute_rules(plu_list):
    """make rules to sub all plus with all other plus (n^2 length)"""
    all_combos = list(itertools.product(plu_list, plu_list))
    assert(len(all_combos) == len(plu_list)**2)
    to_ret = []
    for combination in all_combos:
        to_ret.append(NoisyProduction(Nonterminal("T_"+ combination[0].strip()), [combination[1].strip()]))

    return to_ret

def split_rules(plu_list):
    """make rules to sub all plus with combos of any 2 plus (n^3 length)"""
    all_combos = list(itertools.product(plu_list, plu_list))
    to_ret = []
    for plu in plu_list:
        for combination in all_combos:
            to_ret.append(NoisyProduction(Nonterminal("T_" + plu.strip()), [combination[0].strip(), combination[1].strip()]))
    return to_ret

def map_phone_to_plu(phone_lines, randomize):
    phone_dict = defaultdict()
    train_lines = []
    for i,line in enumerate(phone_lines):
        to_write_train = " ".join(line)
        # to_write_test = truth_lines[i]
        for phone in line:
            phone = phone.strip()
            if re.search('[()*]', phone) is not None:
                phone = "\\" + phone
            try:
                value = phone_dict[phone]
                sent = numpy.random.random()
                if sent <= randomize:
                    # make a new one even though it's in there already
                    value = "".join([str(x) for x in numpy.random.randint(0,9,3)])
                    assert(len(value)==3)
            except KeyError:
                value = "".join([str(x) for x in numpy.random.randint(0,9,3)])
                assert(len(value)==3)
                phone_dict[phone] = value
            to_write_train = re.sub(phone, str(value), to_write_train)
            # with open("mapping", "a+") as f1:
            #     f1.write(",".join(["".join(line).strip(), phone.strip(), value.strip() + "\n"]))
        train_lines.append(to_write_train)

    # with open("truth.dat", "w") as f1:
    #     [f1.write(x) for x in test_lines] 

    return train_lines

def get_used(lines):
    used = set()
    for line in lines:
        splitline = line.split(" ")
        used |= set(splitline)
    return used


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
                        Char -> "000"
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

    with open("brent-phone/train.dat") as f1:
        phone_lines = f1.readlines()

    print("mapping phones")
    with open("brent-phone/train.dat") as f1:
        train_lines= f1.readlines()
    # train_lines = map_phone_to_plu(phone_lines, .1)
    print("done mapping")
    # while i < 100:
    while i < 10:


        # debugging
        grammar_file = open("grammar_debug","w")
        # ten_sents, used = generate_10()
        print("going from ", i*10, " to ", (i+1)*10)
        ten_sents = train_lines[i*10:(i+1)*10]
        print(ten_sents)
        used = get_used(ten_sents)
        print("got sentences, used")

        adagram_inferencer._terminals |= used
        # add the char -> T_xx rules
        print("adding base rules")
        production_list = [Production(Nonterminal("Char"), [Nonterminal("T_{}".format(pre_terminal))]) for pre_terminal in used]

        # add the T_xx -> terminal rules
        production_list += [Production(Nonterminal("T_{}".format(pre_terminal)), [pre_terminal]) for pre_terminal in used]
        # add the substitution rules 
        print("adding sub rules")
        plu_sub_rules = substitute_rules(adagram_inferencer._terminals)
        assert(len(plu_sub_rules) == len(adagram_inferencer._terminals)**2)
        production_list += plu_sub_rules
        # add the combination rules
        print("adding split rules")
        plu_comb_rules = split_rules(adagram_inferencer._terminals)
        assert(len(plu_comb_rules) == len(adagram_inferencer._terminals)**3)
        production_list += plu_comb_rules

        print("updating grammar")
        [grammar_file.write(str(x)+'\n') for x in production_list]
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


