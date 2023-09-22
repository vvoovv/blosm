import re, os

# NOTE: All line numbers in this class refer to a first line having the number zero.
# Editors normally start with line numbe 1, so error messages have to correct for this.

class PML_Preprocessor():
    def __init__(self, rootDir):
        """
        Args:
            rootDir (str): Root directory to use for includes <@include>
                if the path in <@include> starts with '/'
        """
        self.rootDir = rootDir
        self.incl_patt = re.compile('@include.\"')
        self.quot_patt = re.compile('\"(.*?)\"')
        self.cmnt_patt = re.compile('//.*\n?')
        self.outLineNr = 0
        self.lineNrTrack = {}
        self.includeStack = []
        self.stream = ''

    def process(self, path):
        self.includeStack.append(path)
        self.process_includes(path)

    def isIncludeStatement(self, line):
        isInclude = False
        syntaxError = False
        path = None
        match = self.incl_patt.search(line)
        if match:
            isInclude = True
            inQuotations = self.quot_patt.search(line)
            if inQuotations:
                path = inQuotations[0][1:-1]
            else:
                syntaxError = True
        else:
            path = None
        return isInclude, syntaxError, path

    def process_includes(self, path):
        with open(path) as fp:
            self.lineNrTrack[self.outLineNr] = (0,path)
            for localLine, line in enumerate(fp):
                # remove single line comments to avoid commented @include,
                # but keep the line to simplify line counting
                line = re.sub(self.cmnt_patt, '\n', line)
                
                if line != '\n':
                    # check for @include and include lines if required
                    isInclude, syntaxError, inclPath = self.isIncludeStatement(line)
                    if isInclude:
                        if syntaxError:
                            errorText = 'Syntax error in file {file} on line {line}, col {col}: {msg}'.format(
                            file = path, line = localLine+1, col = 0, msg = line)
                            raise Exception(errorText)
                        else:
                            # if the first character of path is '/', add path to the root directory
                            if inclPath[0] == '/':
                                inclPath = os.path.join(
                                    self.rootDir,
                                    os.path.join(*inclPath[1:].split('/'))
                                )
                            # check for recursive inclusion
                            if inclPath in self.includeStack:
                                errorText = 'Error in file {file} on line {line}, col {col}: {msg}'.format(
                                file = path, line = localLine+1, col = 0, msg = 'attempt to @include recursively')
                                raise Exception(errorText)
                            else:
                                inclPath = os.path.join(
                                    os.path.dirname(path),
                                    inclPath
                                )
                                self.includeStack.append(inclPath)
                                # get all lines from included file
                                self.process_includes(inclPath)
                                # replace @include line by empty line to simplify line counting
                                line = "\n"
                                self.lineNrTrack[self.outLineNr] = (localLine,path)
                                self.includeStack.pop()
                
                # push line in output stream
                self.stream += line

                self.outLineNr += 1

    def getStream(self):
        return self.stream

    def getLineNrTrack(self):
        return self.lineNrTrack

    def trackDownLineNr(self, lineNr):
        # The antlr4 error message is already corrected for first line numbber equal to one.
        # find largest line number  in lineNrTrack smaller than lineNr
        index = max( line for line, content in self.lineNrTrack.items() if line < lineNr )
        locaLineNr = self.lineNrTrack[index][0]+(lineNr-index)
        localFile = self.lineNrTrack[index][1]
        return localFile, locaLineNr
