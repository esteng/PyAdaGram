import os
import re
import sys
from collections import defaultdict
import numpy.random

HEADER= """% non-terminals
Words -> Word
Words -> Word Words

% adapted non-terminals
@ Word 1500 100 0

Word -> Chars
Chars -> Char
Chars -> Char Chars

% terminals"""

def read_file(path):
    with open(path) as f1:
        return f1.readlines()

def split_phones(phone_lines):
    phone_lines = [re.sub('([0-9])', '\g<0>!', x) for x in phone_lines]
    return [x.split(" ") for x in phone_lines]

def write_grammar(used, gram_path):
    with open(gram_path, 'w') as f1:
        f1.write(HEADER)
        f1.write("\n")
        

        for char in used:
            f1.write('Char -> "{}"\n'.format(char))

def map_phone_to_plu(phone_lines, truth_lines, randomize):
    phone_dict = defaultdict()
    train_lines = []
    test_lines = []
    used = set()

    for i,line in enumerate(phone_lines):
        to_write_train = " ".join(line)
        to_write_test = truth_lines[i]
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
             
            used |= {value}  
            to_write_train = re.sub(phone, str(value), to_write_train)
            with open("mapping", "a+") as f1:
                f1.write(",".join(["".join(line).strip(), phone.strip(), value.strip() + "\n"]))

            to_write_test = re.sub(phone, str(value), to_write_test)
        train_lines.append(to_write_train)
        test_lines.append(to_write_test)
    with open("train.dat", "w") as f1:
        [f1.write(x) for x in train_lines]
    with open("truth.dat", "w") as f1:
        [f1.write(x) for x in test_lines] 
    return used

def map_back_to_phone(mapping, test_results):
    pass


if __name__ == '__main__':
    phone_lines = split_phones(read_file("/Users/elias/PyAdaGram/brent-phone/train.dat"))
    truth_lines = [" ".join(x) for x in split_phones(read_file("/Users/elias/PyAdaGram/brent-phone/truth.dat"))]
    used = map_phone_to_plu(phone_lines, truth_lines, .01)
    write_grammar(used, "/Users/elias/PyAdaGram/example_nums/unigram.grammar")


