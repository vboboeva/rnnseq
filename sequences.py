import numpy as np
import itertools
from itertools import permutations
from itertools import combinations

# '''
# Adapted from
# https://stackoverflow.com/questions/10244180/python-generating-integer-partitions
# '''
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

def findStructures(L, whichm):

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
    Get all the unique combinations
    '''
    for i, comp_list in enumerate(patterns_by_m):
        patterns_by_m[i] = np.unique(np.sort(np.array(comp_list), axis=1)[:,::-1], axis=0)

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
            # print("\n\nnumber of unique letters = ", m+1)
            # print(templates_list)
            # continue
            structures_list = []
            for template in templates_list:
                # print("\ntemplate = ", template)
                structures_by_template = []
                for perm in permutations(list(template)):
                    perm_ = "".join([s for s in perm])
                    if perm_ not in structures_by_template:
                        structures_by_template.append(perm_)
                # print(structures_by_template)
                structures_list.append(structures_by_template)
            # print(40*'--')
            structures_by_m.append(structures_list)
    # print(40*'*')
    return structures_by_m


if __name__ == "__main__":

    import string


    # Length of sequence
    L = 5
    # Number of elements
    whichm = 2
    # Length of alphabet used
    L_alph = 10
    alphabet = list(string.ascii_lowercase)[:L_alph]
    print(alphabet)
    np.savetxt('input/alphabet.txt',alphabet, fmt='%s')


    structures=findStructures(L, whichm)
    # print(structures)
    structures = [item for sublist in structures for item in sublist]
    # print(structures)
    structures = [item for sublist in structures for item in sublist]
    # print(structures)
    structures=np.array(structures)
    np.savetxt('input/structures.txt',structures, fmt='%s')

    unique=list(set(structures[0]))
    # print(unique)
    # print(40*'*')

    # Here we would like to make many different examples with the same underlying structure

    # remove the letters already present in the sequences
    letters=alphabet
    for i in range(len(unique)):
        letters.remove(unique[i])

    list_combinations=list(itertools.combinations(alphabet, whichm))
    # list_combinations=list(itertools.combinations('abcdefghij', whichm))

    # loop over the individual structures
    for structure in structures:
        possibilities=[]

        # loop over all combinations 
        for comb in list_combinations:
            print('structure', structure)
            print('comb=', comb)
            newseq=np.array(list(structure))
            _, id_list = np.unique(newseq, return_index=True)
            symbols_list = [newseq[idx] for idx in sorted(id_list)]

            print("symbols_list =", symbols_list)

            pos_symbols = [np.where(newseq == sym)[0] for sym in symbols_list]

            print("pos_symbols =", pos_symbols)

            for i, pos in enumerate(pos_symbols):
                print(i, comb[i], pos)
                newseq[pos] = comb[i]

            print('newseq after', newseq)

            possibilities.append(newseq)
            print(20*'--')


        print(len(possibilities))
        np.savetxt('input/%s.txt'%structure, possibilities, fmt='%s')