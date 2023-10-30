import numpy as np
import itertools
from itertools import permutations
from itertools import combinations
import string


'''
Adapted from
https://stackoverflow.com/questions/10244180/python-generating-integer-partitions
'''
def ruleGen(n, m):
    """
    Generates all interpart restricted compositions of n with first part
    >= m. See Kelleher 2006, 'Encoding partitions as ascending compositions'
    chapters 3 and 4 for details.
    """
    a = [0 for i in range(n + 1)]
    k = 1
    a[0] = m - 1
    a[1] = n - m + 1
    while k != 0:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while 1 <= y:
            a[k] = x
            x = 1
            y -= x
            k += 1
        a[k] = x + y
        yield a[:k + 1]


def replace_symbols (sequence, symbols):
    newseq=np.array(list(sequence))

    n_sym_seq = len(np.unique(newseq))
    n_sym_repl = len(np.array(list(symbols)))

    assert n_sym_seq == n_sym_repl, \
        f"Trying to replace {n_sym_seq} symbols in sequence "+ \
        f"with {n_sym_repl} symbols"

    _, id_list = np.unique(newseq, return_index=True)

    symbols_list = [newseq[idx] for idx in sorted(id_list)]
    # print('symbols_list =', symbols_list)
   
    pos_symbols = [np.where(newseq == sym)[0] for sym in symbols_list]
    # print('pos_symbols =', pos_symbols)

    for i, pos in enumerate(pos_symbols):
        # print(i, symbols[i], pos)
        newseq[pos] = symbols[i]

    return "".join(list(newseq))


def findStructures(alphabet, L, whichm):

    patterns_by_m = []
    for m in range(L):
        patterns_by_m.append([])

    '''
    Find composition of L, i.e. all lists of positive integers
    whose sum is L. Group them by length: the length is going to be
    the number of different symbols in a sequence
    '''
    for i, comp in enumerate(ruleGen(L,1)):
        patterns_by_m[len(comp)-1].append(comp)

    '''
    Generate all different templates for each value of 
    different symbols used (m)
    '''
    templates_by_m = []
    for i, comp_list in enumerate(patterns_by_m):
        # print("\nm = ", i+1)
        templates_list = []
        for comp in comp_list:
            template = ""
            for j, n in enumerate(comp):
                template += n*alphabet[j]
            templates_list.append(template)
        templates_by_m.append(templates_list)
        # print(templates_list)


    '''
    Obtain all different structures (up to replacement of symbols)
    from the templates
    '''
    return_m=[]
    structures_by_m = []
    for m, templates_list in enumerate(templates_by_m):
        if m+1 == whichm:
            structures_list = []
            for template in templates_list:
                '''
                Find the unique structures by permuting the symbols
                in `template` and storing only the unique ones -- i.e.
                those that cannot be obtained by others via in-place
                replacement of symbols.
                '''
                for perm in permutations(list(template)):
                    perm_ = "".join([s for s in perm])
                    perm_ = replace_symbols(perm_, alphabet[:whichm])
                    if perm_ not in structures_list:
                        structures_list.append(perm_)
            structures_by_m.append(structures_list)

    return structures_by_m



if __name__ == "__main__":

    from scipy.special import binom, factorial
    
    # Length of sequence
    L = 3
    # Number of elements
    whichm = 2
    # Length of alphabet used
    L_alph = 5
    alphabet = list(string.ascii_lowercase)[:L_alph]
    # print(alphabet)
    np.savetxt('input/alphabet.txt',alphabet, fmt='%s')


    structures=findStructures(alphabet, L, whichm)
    structures = [item for sublist in structures for item in sublist]
    structures=np.array(structures)
    np.savetxt('input/structures_L%d_m%d.txt'%(L, whichm),structures, fmt='%s')

    unique=list(set(structures[0]))
    # print(unique)
    # print(40*'*')

    # Here we would like to make many different examples with the same underlying structure

    # remove the letters already present in the sequences
    letters=alphabet
    # for i in range(len(unique)):
    #     letters.remove(unique[i])

    # all permutations of m letters in the alphabet
    list_permutations=list(itertools.permutations(alphabet, whichm))

    # for perm in list_permutations:
    #     print("".join(list(perm)))

    # print(len(list_permutations))



    # loop over the individual structures
    for structure in structures:
        possibilities=[]
        print('structure =', structure)

        # loop over all permutations 
        for perm in list_permutations:
            print('perm =', perm)
            newseq=replace_symbols(structure, perm)

            possibilities.append(newseq)
            # print(20*'--')


        print(structure)
        np.savetxt('input/%s.txt'%structure, possibilities, fmt='%s')