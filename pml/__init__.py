from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
from .pml_grammar.pmlLexer import pmlLexer
from .pml_grammar.pmlParser import pmlParser
from .PythonListener import PythonListener
from .ExceptionManagement import ParserExceptionListener


class PML:
    
    def __init__(self, pmlFilepath):
        self.pmlFilepath = pmlFilepath
    
    def getPythonCode(self):
        inputStream = FileStream(self.pmlFilepath)
        lexer = pmlLexer(inputStream)
        stream = CommonTokenStream(lexer)
        parser = pmlParser(stream)
        
        parser.removeErrorListeners()
        exceptionListener = ParserExceptionListener()
        parser.addErrorListener(exceptionListener)
        
        tree = parser.styles()
        translator = PythonListener()
        walker = ParseTreeWalker()
        walker.walk(translator, tree)
        return translator.getCode()