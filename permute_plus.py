import itertools






def substitute_rules(plu_list, grammar_file):
    all_combos = list(itertools.product(plu_list, plu_list))
    assert(len(all_combos) == len(plu_list)**2)
    with open(grammar_file, 'w') as f1:
        for combination in all_combos:
            f1.write("& T_{} -> {}\n".format(combination[0], combination[1]))

def split_rules(plu_list, grammar_file):
    all_combos = list(itertools.product(plu_list, plu_list))
    with open(grammar_file, 'a') as f1:
        for plu in plu_list:
            print(plu, len(list(all_combos)))
            for combination in all_combos:
                f1.write("& T_{} -> {}\n".format(plu, " ".join([str(x) for x in combination])))

def delete_rules(plu_list, grammar_file):
    pass

if __name__ == '__main__':
    grammar_file = "permute_test.grammar"
    plu_list_int= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,52,53,54,59,60,62,63,66,68,69,82,83,84,85,86,89,91,94,96,99,102,104,105,107,109,111,114,115,117,126,129,132,136,142,147,150,151,154,155,158]
    plu_list = [str(x) + " " for x in plu_list_int]
    substitute_rules(plu_list, grammar_file)
    split_rules(plu_list, grammar_file)