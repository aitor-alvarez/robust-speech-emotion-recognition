'''
From the set of closed frequent patterns, this class filters out patterns by their type: maximal, minimal.
It takes as input a text file with the list of patterns as strings and with their frequency.
Only the maximal subset is returned, but it can be easily adjusted.
'''

class ClosedPatterns:

    def __init__(self, patterns_file, output_file):
        self.patterns = patterns_file
        self.output_file = output_file


    def execute(self):
        patterns = self.read_files()
        closed=[]
        maximal=[]
        index=[]
        max_index=[]
        minimal=[]
        minimal_index=[]
        blocked=[]
        for i in range (0, len(patterns)):
            for k in range (0, len(patterns)):
                if k == i:
                    continue
                else:
                    if self.isSubpattern(patterns[k][0], patterns[i][0]) and patterns[i][1]>1 and patterns[k][1]>1 and patterns[i][1]>patterns[k][1]:
                        if patterns[i][0] not in closed and patterns[i][0] not in maximal:
                            closed.append(patterns[i][0])
                            index.append(i)
                    elif self.isSubpattern(patterns[k][0], patterns[i][0]) and patterns[i][1]>1 and patterns[i][1]<=patterns[k][1] and patterns[i][0] in closed:
                        ind = closed.index(patterns[i][0])
                        closed.remove(patterns[i][0])
                        index.pop(ind)
                        blocked.append(patterns[i][0])
                    elif (not self.isSubpattern(patterns[k][0], patterns[i][0]) and not self.isSubpattern(patterns[i][0], patterns[k][0])) and patterns[i][1]>1 and patterns[i][0] not in maximal and patterns[i][0] not in closed:
                        maximal.append(patterns[i][0])
                        max_index.append(i)
                    elif patterns[i][0] in maximal and self.isSubpattern(patterns[k][0], patterns[i][0]) and patterns[i][1]>1:
                        indx =maximal.index(patterns[i][0])
                        max_index.pop(indx)
                        maximal.remove(patterns[i][0])
                    elif not self.isSubpattern(patterns[k][0], patterns[i][0]) and patterns[i][1]==1 and patterns[i][0] not in maximal and patterns[i][0] not in closed and patterns[i][0] not in minimal:
                        minimal.append(patterns[i][0])
                        minimal_index.append(i)

        self.write_patterns_to_file(maximal)


    def read_files(self):
        p = open(self.patterns, "r")
        p = p.readlines()
        pat = self.parse_patterns(p)
        return pat


    def isSubpattern(self, pattern, sub):
        if len(sub) >= len(pattern):
            return False
        else:
            result = all(elem in pattern for elem in sub)
            return result


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
        for p in patterns:
            file_.write(str(p) + "\n")
        file_.close()