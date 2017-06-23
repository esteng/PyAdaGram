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
from nltk.grammar import read_grammar, standard_nonterm_parser
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
    training_iterations = 1000
    while i < 100:
        ten_sents, used = generate_10()
        adagram_inferencer._terminals |= used
        for terminal in used:
            lhs_node,rhs_nodes = "Char", terminal
            pcfg_production = 'Char -> "{}"'.format(terminal)
            
            adagram_inferencer._pcfg_productions[(lhs_node, rhs_nodes)].add(pcfg_production);
            
            adagram_inferencer._lhs_rhs_to_pcfg_production[(lhs_node, rhs_nodes)] = pcfg_production
            adagram_inferencer._lhs_to_pcfg_production[lhs_node].add(pcfg_production);
            adagram_inferencer._rhs_to_pcfg_production[rhs_nodes].add(pcfg_production);
            if len(rhs_nodes)==1:
                adagram_inferencer._rhs_to_unary_pcfg_production[rhs_nodes[0]].add(pcfg_production);

            adagram_inferencer._gamma_index_to_pcfg_production_of_lhs[lhs_node][len(adagram_inferencer._gamma_index_to_pcfg_production_of_lhs[lhs_node])] = pcfg_production;
            adagram_inferencer._pcfg_production_to_gamma_index_of_lhs[lhs_node][pcfg_production] = len(adagram_inferencer._pcfg_production_to_gamma_index_of_lhs[lhs_node]);

        topology_order, order_topology = adagram_inferencer._topological_sort();
        
        adagram_inferencer._incremental_build_up = False;
        adagram_inferencer._non_terminal_to_level = topology_order;
        adagram_inferencer._level_to_non_terminal = order_topology;
        
        adagram_inferencer._ordered_adaptor_top_down = [];
        for x in xrange(len(order_topology)):
            for non_terminal in order_topology[x]:
                if non_terminal in adagram_inferencer._adapted_non_terminals:
                    adagram_inferencer._ordered_adaptor_top_down.append(non_terminal);
        adagram_inferencer._ordered_adaptor_down_top = adagram_inferencer._ordered_adaptor_top_down[::-1];
        # for iteration in range(training_iterations):
        # pick up again here!

        clock_iteration = time.time();
        number_of_processes = 1;
        print(ten_sents)
        clock_e_step, clock_m_step = adagram_inferencer.learning(ten_sents, number_of_processes);

        if (i+1)%snapshot_interval==0:
            adagram_inferencer.export_adaptor_grammar(os.path.join(output_directory, "adagram-" + str((iteration+1))))

        if (i+1) % 1000==0:
            snapshot_clock = time.time() - snapshot_clock;
            print 'Processing 1000 mini-batches take %g seconds...' % (snapshot_clock);
            snapshot_clock = time.time()
        clock_iteration = time.time()-clock_iteration;
        print 'E-step, M-step and iteration %d take %g, %g and %g seconds respectively...' % (adagram_inferencer._counter, clock_e_step, clock_m_step, clock_iteration);
    
        i+=1    
    adagram_inferencer.export_adaptor_grammar(os.path.join(output_directory, "adagram-" + str(adagram_inferencer._counter+1)))
    cpickle_file = open(os.path.join(output_directory, "model-%d" % (iteration+1)), 'wb');
    cPickle.dump(adagram_inferencer, cpickle_file);
    cpickle_file.close();
    
    training_clock = time.time()-training_clock;
    print 'Training finished in %g seconds...' % (training_clock);


