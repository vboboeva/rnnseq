import numpy as np
from itertools import permutations
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
        # we cannot have m > |alphabet|
        # exclude these combinations
        m = i+1
        if m > len(alphabet):
            break
        print("\nm = ", m)
        templates_list = []
        for comp in comp_list:
            template = ""
            print("\t", comp)
            for j, n in enumerate(comp):
                # print(j, n)
                template += n*alphabet[j]
            # exit()
            templates_list.append(template)
        templates_by_m.append(templates_list)
        # print(templates_list)

    '''
    Obtain all different types (up to replacement of symbols)
    from the templates
    '''
    return_m=[]
    types_by_m = []
    for m, templates_list in enumerate(templates_by_m):
        if m+1 == whichm:
            types_list = []
            for template in templates_list:
                '''
                Find the unique types by permuting the symbols
                in `template` and storing only the unique ones -- i.e.
                those that cannot be obtained by others via in-place
                replacement of symbols.
                '''
                for perm in permutations(list(template)):
                    perm_ = "".join([s for s in perm])
                    perm_ = replace_symbols(perm_, alphabet[:whichm])
                    if perm_ not in types_list:
                        types_list.append(perm_)
            types_by_m.append(types_list)

    return types_by_m

if __name__ == "__main__":
    
    # Length of sequence
    L = 4
    # Number of elements
    whichm = 2
    # Length of alphabet used
    alpha = 10
    alphabet = list(string.ascii_lowercase)[:alpha]
    print(alphabet)
    np.savetxt('input/alphabet.txt',alphabet, fmt='%s')


    types=findStructures(alphabet, L, whichm)
    types = [item for sublist in types for item in sublist]
    types=np.array(types)
    np.savetxt('input/types_L%d_m%d.txt'%(L, whichm),types, fmt='%s')

    # Here we would like to make many different examples with the same underlying type_

    # # all permutations of m letters in the alphabet
    # list_permutations=list(itertools.permutations(alphabet, whichm))

    # # loop over the individual types
    # for type_ in types:
    #     possibilities=[]
    #     print('type_ =', type_)

    #     # loop over all permutations 
    #     for perm in list_permutations:
    #         print('perm =', perm)
    #         newseq=replace_symbols(type_, perm)

    #         possibilities.append(newseq)
    #         # print(20*'--')
            
    #     # print(type_)
    #     print(possibilities)
    #     np.savetxt('input/%s.txt'%type_, possibilities, fmt='%s')