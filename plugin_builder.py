import sys, os, re

_reImport = re.compile(r"\s*import\s+(.+)")
_reFrom = re.compile(r"\s*from\s+([\w.]+)\s+.+")

class PluginBuilder:
    
    def __init__(self, pluginPath):
        # for the development version of the plugin:
        pluginFullPathDev = os.path.realpath(pluginPath)
        self.pluginFileNameDev = os.path.basename(pluginFullPathDev)
        self.pluginDir = os.path.dirname(pluginFullPathDev)
        # for the release version of the plugin:
        pluginFileNameRel = self.pluginFileNameDev[:-7] + ".py" # cut off _dev suffix from pluginFileName1
        self.pluginFullPathRel = os.path.join(self.pluginDir, pluginFileNameRel)

    def build(self):
        with open(self.pluginFullPathRel, "w") as output:
            self.output = output
            self.writeHeader()
            self.writeLocalModule(self.pluginFileNameDev[:-3])

    def writeLocalModule(self, module):
        moduleFullPath = os.path.join(self.pluginDir, module.replace(".", "/")+".py")
        if not os.path.exists(moduleFullPath): return True
        
        with open(moduleFullPath, "r") as input:
            modulesForRemoval = []
            for line in input:
                writeLine = True
                # checking if the line is of type <import module1, module2>
                matchResult = _reImport.match(line)
                if matchResult:
                    # get a list of modules
                    modules = [m.strip() for m in matchResult.group(1).split(",")]
                    for m in modules:
                        writeLine = self.writeLocalModule(m)
                        if not writeLine:
                            # it's local module, so remove all reference to it like module.someModuleFunction(...)
                            modulesForRemoval.append(m)
                else:
                    # checking if the line is of type <from module import ....>
                    matchResult = _reFrom.match(line)
                    if matchResult:
                        writeLine = self.writeLocalModule(matchResult.group(1))
                    else:
                        # skip sys.path.append(...) commands
                        if line.find("sys.path.append") >= 0:
                            writeLine = False
                if writeLine:
                    # remove all reference to local modules from modulesForRemoval like module.someModuleFunction(...)
                    for m in modulesForRemoval:
                        line = re.sub(r"(\s+)"+m+"\.", "\g<1>", line)
                    self.output.write(line)
            self.output.write(os.linesep)
        return False

    def writeHeader(self):
        print("# This is the release version of the plugin file %s" % self.pluginFileNameDev, file=self.output)
        print("# If you would like to make edits, make them in the file %s and the other related modules" % self.pluginFileNameDev, file=self.output)
        print("# To create the release version of %s, execute:" % self.pluginFileNameDev, file=self.output)
        print("# python plugin_builder.py %s" % self.pluginFileNameDev, file=self.output)
    
if __name__ == "__main__":
    PluginBuilder(sys.argv[1]).build()