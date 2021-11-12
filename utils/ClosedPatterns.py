'''
From the set of closed frequent patterns, this class filters out patterns by their type: maximal, minimal.
It takes as input a text file with the list of patterns as strings and with their frequency.
Only the maximal subset is returned.
'''

class ClosedPatterns:

    def __init__(self, patterns_file, output_file):
        self.patterns = patterns_file
        self.output_file = output_file


    def execute(self):
        patterns = self.read_files()
        closed = []
        maximal = []
        index = []
        max_index = []
        minimal = []
        minimal_index = []
        blocked = []
        max_freq=[]
        for i in range(0, len(patterns)):
            for k in range(0, len(patterns)):
                if k == i:
                    continue
                else:
                    if self.isSubpattern(patterns[k][0], patterns[i][0]) and patterns[i][1] > 1 and patterns[k][1] > 1 and patterns[i][1] > patterns[k][1]:
                        if patterns[i][0] not in closed and patterns[i][0] not in maximal:
                            closed.append(patterns[i][0])
                            index.append(i)
                    elif self.isSubpattern(patterns[k][0], patterns[i][0]) and patterns[i][1] > 1 and patterns[i][1] <=  patterns[k][1] and patterns[i][0] in closed:
                        ind = closed.index(patterns[i][0])
                        closed.remove(patterns[i][0])
                        index.pop(ind)
                        blocked.append(patterns[i][0])
                    elif (not self.isSubpattern(patterns[k][0], patterns[i][0]) and not self.isSubpattern(patterns[i][0],patterns[k][ 0])) and patterns[i][0] not in maximal and patterns[i][0] not in closed:
                        maximal.append(patterns[i][0])
                        max_freq.append(patterns[i][1])
                        max_index.append(i)
                    elif patterns[i][0] in maximal and self.isSubpattern(patterns[k][0], patterns[i][0]) and patterns[i][1] > 1:
                        indx = maximal.index(patterns[i][0])
                        max_index.pop(indx)
                        maximal.remove(patterns[i][0])
                    elif not self.isSubpattern(patterns[k][0], patterns[i][0]) and patterns[i][1] == 1 and patterns[i][
                        0] not in maximal and patterns[i][0] not in closed and patterns[i][0] not in minimal:
                        minimal.append(patterns[i][0])
                        minimal_index.append(i)
        self.write_patterns_to_file(maximal)


    def read_files(self):
        p = open(self.patterns, "r")
        p = p.readlines()
        pat = self.parse_patterns(p)
        return pat


    def isSubpattern(self, pattern, sub):
        if sub == pattern:
            return True
        else:
            p = ''.join(pattern)
            s = ''.join(sub)
            if p.find(s) !=-1:
                return True
            else:
                return False


    def parse_patterns(self, p):
        patterns = []
        for el in p:
            out = el[:el.find (']') + 1]
            out = out.replace ('[', '').replace(']', '').replace("'", '').replace(',', ' ')
            out = out.split()
            patterns.append((out, int(el[el.find(']') + 1:].replace('\n',''))))
        return patterns


    def write_patterns_to_file(self, patterns, freq):
        file_ = open(self.output_file, 'a')
        for p in range(0, len(patterns)):
            file_.write(str(patterns[p])+str(freq[p]) + "\n")
        file_.close()


class UniquePatterns:
    # Find unique patterns in a class when comparing to a reference (reference_class_file)
    # pass 2 pattern files and get the list of unique patterns written to a new text file.
    def __init__(self, reference_class_file, class_file):
        self.ref = reference_class_file
        self.compare = class_file

    def get_unique_patterns(self):
        ref_pat = self.read_files(self.ref)
        refp = [r[0] for r in ref_pat ]
        compare_pat = self.read_files(self.compare)
        comp = [c[0] for c in compare_pat]
        unique = []
        freq=[]
        sub=[]
        for c in range(0, len(comp)):
            if comp[c] in refp:
                pass
            elif comp[c] not in unique and comp[c] not in sub:
                unique.append(compare_pat[c][0])
                freq.append(compare_pat[c][1])
        for u in unique:
            for k in refp:
                if ClosedPatterns.isSubpattern('', u, k) or ClosedPatterns.isSubpattern('', k, u):
                    if u in unique:
                        indx = unique.index(u)
                        unique.pop(indx)
                        freq.pop(indx)
        self.write_patterns_to_file(unique, freq)
        print(str(len(unique))+" unique patterns extracted")


    def read_files(self, patterns):
        p = open(patterns, "r")
        p = p.readlines()
        pat = ClosedPatterns.parse_patterns('', p)
        return pat

    def write_patterns_to_file(self, patterns, freq):
        filename = self.compare.replace('.txt', '')+'_unique.txt'
        file_ = open(filename, 'a')
        for p in range(0, len(patterns)):
            file_.write(str(patterns[p])+str(freq[p]) + "\n")
        file_.close()