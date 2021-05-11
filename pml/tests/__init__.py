"""
This file is also needed for the correct work of relative imports in the tests. For example:
from . import getPythonCode
from .. import PML
"""

from .. import PML
import os, ast, tempfile


def compare(output, referenceOutput):
    referenceOutput = "styles = [%s]" % referenceOutput
    return ''.join(output.split()) == ''.join(referenceOutput.split())


def getPythonCode(pmlString, rootDir=''):
    with tempfile.TemporaryDirectory() as tmpDir:
        tmpFileName = os.path.join(tmpDir, "test.pml")
        with open(tmpFileName, 'w') as tmpFile:
            tmpFile.write(pmlString)
        pythonCode = PML(tmpFileName, rootDir).getPythonCode()
    return pythonCode


def makeTest(input, referenceOutput):
    output = getPythonCode(input)
    ast.parse(output)
    assert compare(output, referenceOutput)