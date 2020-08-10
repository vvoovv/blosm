import sys
import os
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from pml_grammar.pmlLexer import pmlLexer
from pml_grammar.pmlParser import pmlParser
from PythonListener import PythonListener
from ExceptionManagement import ParserExceptionListener
from PML_Preprocessor import PML_Preprocessor
from ExceptionManagement import ParserException

def main(argv):
    # argv[2] is the path to the folder where the asset packages
    # are stored. Do not add '/' at end.
    preprocessor = PML_Preprocessor(argv[2])
    hadErrors = False
    errorText = ''
    try:
        preprocessor.process(argv[1])
    except Exception as e:
        errorText = str(e)
        hadErrors = True

    if not hadErrors:
        input_stream = InputStream(preprocessor.getStream())
        lexer = pmlLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = pmlParser(stream)

        parser.removeErrorListeners()
        exceptionListener = ParserExceptionListener()
        parser.addErrorListener( exceptionListener )

        try:
            tree = parser.styles()
        except ParserException as e:
            line,col,msg = e.errParams()
            localFile,localLine = preprocessor.trackDownLineNr(line)
            errorText = 'Error in file {file} on line {line}, col {col}: {msg}'.format(
            file = localFile, line = localLine, col = col, msg = msg)
            hadErrors = True
        except Exception as e:
            errorText = str(e)
            hadErrors = True

        if not hadErrors:  
            translator = PythonListener()
            walker = ParseTreeWalker()
            walker.walk(translator, tree)
            sys.stdout.write( translator.getCode() )

    if hadErrors:
            sys.stdout.write(errorText)

if __name__ == '__main__':
    main(sys.argv)