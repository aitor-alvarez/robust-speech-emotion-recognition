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


    def isSubpattern(pattern, sub):
        if pattern.find(sub) != -1:
            return True
        else:
            return False


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
            for c in comp:
                if u.find(c) != -1 and c.find(u) != -1 and c not in output:
                    output.append(c)

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
