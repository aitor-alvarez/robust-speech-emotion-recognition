'''
From the set of closed frequent patterns, the class MaximalPatterns extracts maximal patterns.
It takes as input a text file with the list of patterns as strings.
Only the maximal subset is returned.
'''

from itertools import combinations

class MaximalPatterns:

    def __init__(self, patterns_file, output_file):
        self.patterns = patterns_file
        self.output_file = output_file


    def execute(self):
        patterns = self.read_files()
        pstr = [''.join(p[0]) for p in patterns]
        pstr.sort(key=len, reverse=True)
        not_max=[]
        for p, k in combinations(pstr, 2):
            if p.find(k) != -1:
                not_max.append(k)
        output = [p for p in pstr if p not in not_max]
        self.write_patterns_to_file(output)


    def read_files(self):
        p = open(self.patterns, 'r')
        p = p.readlines()
        pat = self.parse_patterns(p)
        return pat


    def parse_patterns(self, p):
        patterns = []
        for el in p:
            out = el[:el.find (']') + 1]
            out = out.replace ('[', '').replace(']', '').replace("'", '').replace(',', ' ')
            out = out.split()
            patterns.append((out, int(el[el.find(']') + 1:].replace('\n',''))))
        return patterns



    def write_patterns_to_file(self, patterns):
        file_ = open(self.output_file, 'a')
        for p in range(0, len(patterns)):
            file_.write(str(patterns[p]) + "\n")
        file_.close()


class UniquePatterns:
    # Find unique patterns in a class when comparing to a reference (reference_class_file)
    # pass 2 pattern files and get the list of unique patterns written to a new text file.
    def __init__(self, reference_class_file, class_file):
        self.ref = reference_class_file
        self.compare = class_file

    def get_unique_patterns(self):
        ref = self.read_files(self.ref)
        comp = self.read_files(self.compare)
        unique = [c for c in comp if c not in ref]
        output = []
        for u in unique:
            for r in ref:
                if u.find(r) == -1 and r.find(u) == -1 and u not in output:
                    output.append(u)
                elif (u.find(r) != -1 or r.find(u) != -1) and u in output:
                    output.remove(u)

        self.write_patterns_to_file(output)
        print(str(len(output))+" unique patterns extracted")


    def read_files(self, patterns):
        p = open(patterns, 'r')
        p = p.readlines()
        return p

    def write_patterns_to_file(self, patterns):
        filename = self.compare.replace('.txt', '')+'_unique.txt'
        file_ = open(filename, 'a')
        for p in range(0, len(patterns)):
            file_.write(str(patterns[p]))
        file_.close()
