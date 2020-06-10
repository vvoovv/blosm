import sys
import os
from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
from pml_grammar.pmlLexer import pmlLexer
from pml_grammar.pmlParser import pmlParser
from PythonListener import PythonListener
from ExceptionManagement import ParserExceptionListener

def main(argv):
    # don't know if this works for all OS
    input_stream = FileStream(argv[1])
    lexer = pmlLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = pmlParser(stream)

    parser.removeErrorListeners()
    exceptionListener = ParserExceptionListener()
    parser.addErrorListener( exceptionListener )

    # error management
    hadSyntaxErrors = False
    try:
        tree = parser.styles()
    except Exception as e:
        errorText = str(e)
        hadSyntaxErrors = True

    if not hadSyntaxErrors:  
        translator = PythonListener()
        walker = ParseTreeWalker()
        walker.walk(translator, tree)
        sys.stdout.write( translator.getCode() )
    else:
        sys.stdout.write(errorText)

if __name__ == '__main__':
    main(sys.argv)