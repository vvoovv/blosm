# patched ErrorListener from antlr4
class ParserExceptionListener(object):
    def syntaxError(self, recognizer, offendingSymbol, line, col, msg, e):
        raise ParserException(line, col, msg)

    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        pass

    def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
        pass

    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        pass


class ParserException(Exception):    
    def __init__(self, line, col, msg):
        self.line = line
        self.col = col
        self.msg = msg

    def errParams(self):
        return self.line, self.col, self.msg

    def __str__(self):
        return 'Error on line {line}, col {col}: {msg}'.format(
            line = self.line, col = self.col, msg = self.msg)