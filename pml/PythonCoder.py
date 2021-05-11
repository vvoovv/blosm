import re
from Dictionaries import Dictionaries

class PythonCoder():
    def __init__(self):
        self.dictionary = Dictionaries()
        self.code = ""
        self.blockCommaStack = []
        self.elementCommaStack = []
        self.attribCommaStack = []
        self.alterCommaStack = []
        self.condCommaStack = []
        self.indents = 0
        self.smoothContext = False
        self.alternativesContext = False
        self.conditionContext = False
        self.conditionalContext = False
        self.spec_condition = []

    def getCode(self):
        return self.code

    def indent(self):
        return (" "*4*self.indents)

    def write(self,text):
        self.code += text

    def literalize(self,text, useUnderscore):
        if text[0] == '"':
            return text
        if useUnderscore:
            literalized = re.sub('("[ _]+")', ' ', re.sub('("")', '"', re.sub('([a-zA-Z]+)', '"\\1"', text) ) )
        else:
            literalized = re.sub('([a-zA-Z_]+[a-zA-Z0-9_]*)', '"\\1"', text) 
        return literalized

    def toCamelCase(self,text):
        return ''.join([ x.capitalize() for x in text.split('_') ])

    def replaceHexColorCode(self, match):
        value = match.group(1)
        if len(value) == 3:  # short group
            value = [str(round(int(c + c, 16)/255.0,3)) for c in value]
        elif len(value) == 6:
            value = [str(round(int(c1 + c2, 16)/255.0,3)) for c1, c2 in zip(value[::2], value[1::2])]
        else:
            raise Exception('Invalid hex number: #' + value)
        return '({}, 1.0)'.format(', '.join(value))

    def replaceRGBColorCode(self,match):
        rgb = match.group(0)
        values = rgb[4:-1].split(',')
        values.append('255')
        return str( tuple( round(c/255.,3) for c in values ) )

    def replaceRGBAColorCode(self,match):
        rgba = match.group(0)
        values = rgba[5:-1].split(',')
        values[3] = str( 255.*float(values[3]) )
        return str( tuple( round(c/255.,3) for c in values ) )

    def replaceColorsInText(self,text):
        # _hex_colour = re.compile(r'#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b')
        _hex_colour = re.compile(r'#([0-9a-fA-F]+|[0-9a-fA-F]+)\b')
        text = _hex_colour.sub(self.replaceHexColorCode, text)

        _rgb_colour = re.compile(r'rgb\(\s*(?:(\d{1,3})\s*,?){3}\)')
        text = _rgb_colour.sub(self.replaceRGBColorCode, text)

        _rgba_colour = re.compile(r'rgba\(\s*(?:(\d{1,3})\s*,?){3},\d+\.\d+\)')
        text = _rgba_colour.sub(self.replaceRGBAColorCode, text)

        for word, initial in self.dictionary.colors.items():
             text = re.sub(r'\b'+word+r'\b',str(initial),text)
        return text
        
    # ------------------------------------------------------------
    # styles
    # : named_block+  EOF            #NAMED
    # | elements EOF                 #UNNAMED
    # ;
    # ------------------------------------------------------------

    def enterNAMED(self):
        self.indents = 1
        self.blockCommaStack.append('')
        self.write('styles = {\n' )

    def exitNAMED(self):
        self.blockCommaStack.pop()
        self.write('\n}')

    def enterUNNAMED(self):
        self.indents = 1
        self.write('styles = [' )

    def exitUNNAMED(self):
        self.write(']')

    # ------------------------------------------------------------
    # named_block
    #     : ('@name' STRING_LITERAL SEMI elements)
    #     ;
    # ------------------------------------------------------------

    def enterNamed_block(self,name):
        self.write(self.blockCommaStack[-1])
        self.write(self.indent()+name + ' : [')
        self.blockCommaStack[-1] = ',\n'
        self.indents += 1
    
    def exitNamed_block(self):
        self.write(self.indent()+']') 

    # ------------------------------------------------------------
    # elements
    #     : element ( element )* 
    #     ;
    # ------------------------------------------------------------

    def enterElements(self):
        self.elementCommaStack.append('\n')
        self.attribCommaStack.append("\n")

    def exitElements(self):
        self.write("\n")
        self.indents -= 1
        self.elementCommaStack.pop()

    # ------------------------------------------------------------
    # element 
    #     : '@'? 'level' (STRUDEL def_name)? spec_conditions* (LPAREN condition RPAREN)? LCURLY attributes RCURLY
    #     | '@'? element_name (STRUDEL def_name)? (LPAREN condition RPAREN)? LCURLY attributes RCURLY
    #     ;
    # ------------------------------------------------------------

    def enterElement_name(self,name):
        txt = self.toCamelCase(name)
        self.write(self.indent()+txt+'(')
        self.indents += 1

    def enterElement(self,txt):
        self.write(self.elementCommaStack[-1])
        self.attribCommaStack.append("\n")  # maybe we have a condition first
        if 'level' in txt:
            self.write(self.indent()+'Level(')
            self.indents += 1

 
    def exitElement(self):
        self.indents -=1
        self.write(self.indent()+')')
        self.elementCommaStack[-1] = ",\n"

    def enterCondition(self):
        self.conditionContext = True
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+'condition = lambda item : ' )
        self.attribCommaStack[-1] = ',\n'

    def exitCondition(self):
        self.conditionContext = False
        self.attribCommaStack[-1] = ',\n'


 
    # ------------------------------------------------------------
    # attributes
    #     : attribute*
    #     ;
    # ------------------------------------------------------------

    def enterAttributes(self):
        pass
 #      self.attribCommaStack.append("\n")

    def exitAttributes(self):
        self.write("\n")
        self.attribCommaStack.pop()

    # ------------------------------------------------------------
    # attribute 
    #     : 'symmetry' COLON sym_expression SEMI
    #     | 'use' COLON use_expression SEMI
    #     | ('faces' | 'sharpEdges') COLON smooth_expression SEMI
    #     | attr_name COLON expression SEMI
    #     | attr_name COLON markup_block  // markup
    #     ;
    # sym_expression
    #     : IDENTIFIER
    #     ;
    # use_expression
    #     : IDENTIFIER (COMMA IDENTIFIER)*
    #     ;
    # smooth_expression
    #     : expression
    #     ;
    # markup_block
    #     : LBRACK elements RBRACK
    #     ;
    # ------------------------------------------------------------

    def enterSym_expression(self,sym):
        self.write(self.attribCommaStack[-1])
        symmetry = self.toCamelCase(sym)
        self.write(self.indent()+'symmetry = symmetry.'+symmetry )
        self.attribCommaStack[-1] = ",\n"

    def enterUse_expression(self,enterSimple_expr):
        self.write(self.attribCommaStack[-1])
        expression = self.literalize(enterSimple_expr,False)
        self.write(self.indent()+'use = (' + expression + ',)' )
        self.attribCommaStack[-1] = ",\n"

    def enterSmooth_expression(self, name):
        self.smoothContext = True
        self.write(self.attribCommaStack[-1])
        self.write(self.indent() + name + ' = ' )
        self.attribCommaStack[-1] = ",\n"

    def exitSmooth_expression(self):
        self.smoothContext = False

    def enterMarkup_block(self):
        self.write(' [')
        self.indents += 1

    def exitMarkup_block(self):
        self.write(self.indent()+']')

    def enterAttr_name(self,name):
        if name == 'class':     # avoid conflict with Python keyword
            name = 'cl'
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+name+' = ')
        self.attribCommaStack[-1] = ",\n"

    # ------------------------------------------------------------
    # expression
    #     :  simple_expr | function | alternatives
    #     ;
    # alternatives
    #     : function (PIPE function)+
    #     ;
    # ------------------------------------------------------------

    def enterAlternatives(self):
        self.alternativesContext = True
        # self.inFunctionStack.append('inFunction')
        self.functionContext = True
        self.write("Value(Alternatives(\n")
        self.alterCommaStack.append("")
        self.indents +=1

    def exitAlternatives(self):
        self.alternativesContext = False
        self.alterCommaStack.pop()
        self.indents -=1
        self.write('\n'+self.indent()+'))')

    # ------------------------------------------------------------
    # function
    #     : 'attr' LPAREN string_literal RPAREN           #ATTR
    #     | 'bldgAttr' LPAREN string_literal RPAREN       #BUILDATTR
    #     | 'random_normal' LPAREN NUMBER RPAREN          #RANDN
    #     | 'random_weighted' nested_list                 #RANDW
    #     | 'if' LPAREN conditional RPAREN function       #COND
    #     | 'use_from' LPAREN IDENTIFIER RPAREN           #USEFROM
    #     | 'per_building' LPAREN function RPAREN         #PERBUILD
    #     | 'rgb' LPAREN NUMBER COMMA NUMBER COMMA NUMBER RPAREN      #RGB
    #     | 'rgba' LPAREN NUMBER COMMA NUMBER COMMA NUMBER COMMA NUMBER RPAREN      #RGBA
    #     | constant                                      #CONST
    #     | nested_list                                   #NESTED
    #     | arith_atom                                    #ARITH
    #     ;
    # ------------------------------------------------------------

    def enterATTR(self,attribute):
        types = self.dictionary.getAttributeTypes(attribute)
        if self.alternativesContext:
            self.write(self.alterCommaStack[-1])
            if len(types) == 1:
                self.write( self.indent()+"FromAttr(" + attribute + ", " + types[0] + ')' )
            else:
                self.write( self.indent()+"FromAttr(" + attribute + ", " + types[0] + '),\n' )
                self.write( self.indent()+"FromAttr(" + attribute + ", " + types[1] + ')' )
            self.alterCommaStack[-1] = ",\n"
        else:
            if len(types) == 1:
                self.write( "Value(FromAttr(" + attribute + ", " + types[0] + '))' )
            else:
                self.write('Value(Alternatives(\n')
                self.indents += 1
                self.write( self.indent()+"FromAttr(" + attribute + ", " + types[0] + '),\n' )
                self.write( self.indent()+"FromAttr(" + attribute + ", " + types[1] + ')\n' )
                self.indents -= 1
                self.write(self.indent()+'))')

    def enterBUILDATTR(self,attribute):
        types = self.dictionary.getAttributeTypes(attribute)
        if self.alternativesContext:
            self.write(self.alterCommaStack[-1])
            if len(types) == 1:
                self.write( self.indent()+"FromBldgAttr(" + attribute + ", " + types[0] + ')' )
            else:
                self.write( self.indent()+"FromBldgAttr(" + attribute + ", " + types[0] + '),\n' )
                self.write( self.indent()+"FromBldgAttr(" + attribute + ", " + types[1] + ')' )
            self.alterCommaStack[-1] = ",\n"
        else:
            if len(types) == 1:
                self.write( "Value(FromBldgAttr(" + attribute + ", " + types[0] + '))' )
            else:
                self.write('Value(Alternatives(\n')
                self.indents += 1
                self.write( self.indent()+"FromBldgAttr(" + attribute + ", " + types[0] + '),\n' )
                self.write( self.indent()+"FromBldgAttr(" + attribute + ", " + types[1] + ')\n' )
                self.indents -= 1
                self.write(self.indent()+'))')

    # ------------------------------------------------------------
    # function
    #     : ...
    #     | 'random_normal' LPAREN NUMBER RPAREN          #RANDN
    #     | 'random_weighted' nested_list                 #RANDW
    #     | ...
    # ------------------------------------------------------------

    def enterRANDN(self,value):
        if self.alternativesContext or self.conditionContext:
            self.write(self.alterCommaStack[-1])
            self.write(self.indent()+'RandomNormal( ' + value + ' )')
            self.alterCommaStack[-1] = ",\n"
        else:
            self.write('Value(RandomNormal( ' + value + ' ))')

    def enterRANDW(self,li):
        li = self.replaceColorsInText(li)
        list = self.literalize(li,False)
        if self.alternativesContext or self.conditionContext:
            self.write(self.alterCommaStack[-1])
            self.write(self.indent()+'RandomWeighted( ' + list + ' )')
            self.alterCommaStack[-1] = ",\n"
        else:
            self.write('Value(RandomWeighted( ' + list + ' ))')

    # ------------------------------------------------------------
    # function
    #     : ...
    #     | 'if' LPAREN conditional RPAREN function       #COND
    #     | ...
    # ------------------------------------------------------------

    def enterCOND(self, condition, result):
        self.conditionContext = True
        self.conditionalContext = True
        if self.alternativesContext:
            self.write(self.alterCommaStack[-1])
            self.write( self.indent()+"Conditional(\n" )
            self.indents += 1
            self.write(self.indent()+'lambda item: ' ) 
            self.alterCommaStack.append(',\n')
        else:
            self.write( "Value(Conditional(\n" )
            self.indents += 1
            self.write(self.indent()+'lambda item: ' ) 

    def exitCOND(self):
        self.indents -= 1
        self.write("\n")
        if self.conditionContext:
            if self.alternativesContext:
                self.write(self.indent()+")" )
            else:
                self.write(self.indent()+"))" )
        else:
            self.write(self.indent()+")" )
        self.conditionContext = False
        self.conditionalContext = False

    # ------------------------------------------------------------
    # function
    #     : ...
    #     | 'use_from' LPAREN IDENTIFIER RPAREN           #USEFROM
    #     | 'per_building' LPAREN function RPAREN         #PERBUILD
    #     | ...
    # ------------------------------------------------------------

    def enterUSEFROM(self,ident):
        self.write('useFrom("' + ident + '")')

    def enterPERBUILD(self):
        if self.alternativesContext:
            self.write(self.alterCommaStack[-1])
            self.alterCommaStack[-1] = ''
            self.write(self.indent()+'PerBuilding(\n')
            self.indents += 1
        else:
            self.write('PerBuilding(')

    def exitPERBUILD(self):
        if self.alternativesContext:
            self.indents -= 1
            self.alterCommaStack[-1] = ',\n'
            self.write('\n'+self.indent()+')')
        else:
            self.write(')')

    # ------------------------------------------------------------
    # function
    #     : ...
    #     | 'rgb' LPAREN NUMBER COMMA NUMBER COMMA NUMBER RPAREN      #RGB
    #     | 'rgba' LPAREN NUMBER COMMA NUMBER COMMA NUMBER COMMA NUMBER RPAREN      #RGBA
    #     | ...
    # ------------------------------------------------------------

    def enterRGB(self,rgb):
        expr = self.replaceColorsInText(rgb)
        if self.alternativesContext or self.conditionalContext:
            self.write(self.alterCommaStack[-1])
            self.write( self.indent()+"Constant(" + expr + ')' )
            self.alterCommaStack[-1] = ',\n'
        else:
            self.write('Value(Constant(' + expr + ')' )

    def enterRGBA(self,rgba):
        expr = self.replaceColorsInText(rgba)
        if self.alternativesContext or self.conditionalContext:
            self.write(self.alterCommaStack[-1])
            self.write( self.indent()+"Constant(" + expr + ')' )
            self.alterCommaStack[-1] = ',\n'
        else:
            self.write('Value(Constant(' + expr + ')' )

    # ------------------------------------------------------------
    # function
    #     : ...
    #     | constant                                      #CONST
    #     | nested_list                                   #NESTED
    #     | ...
    # ------------------------------------------------------------

    def enterCONST(self,text):
        text = self.replaceColorsInText(text)

        if self.conditionalContext:
            self.write(',\n')
            if self.smoothContext and text in ('smooth','flat','horizontal','side','all'):
                expr = self.toCamelCase(text)
                self.write( self.indent()+"Constant(smoothness." + expr + ')' )
            else:
                expr = self.literalize(text,False)
                self.write( self.indent()+"Constant(" + expr + ')' )
        elif self.alternativesContext: 
            self.write(self.alterCommaStack[-1])
            expr = self.literalize(text,False)
            self.write( self.indent()+"Constant(" + expr + ')' )
            self.alterCommaStack[-1] = ',\n'
        else:
            expr = self.literalize(text,False)
            self.write(expr)

    def enterNESTED(self, li):
        li = self.replaceColorsInText(li)
        list = self.literalize(li,False)
        if self.alternativesContext:
            self.write(self.alterCommaStack[-1])
            self.write( self.indent()+list )
            self.alterCommaStack[-1] = ',\n'
        else:
            self.write( list )

    def exitINNESTED(self,li):
        list = self.literalize(li,True)
        self.write( list )

    # ------------------------------------------------------------
    # spec_conditions
    #     : LBRACK spec_condition RBRACK  #SPEC_LEVEL
    #     ;
    # spec_condition
    #     : 'roof'                    #SPEC_ROOF
    #     | 'all'                     #SPEC_ALL
    #     | NUMBER COLON NUMBER       #SPEC_FULL_INDX
    #     | NUMBER COLON              #SPEC_LEFT_INDX
    #     | COLON NUMBER              #SPEC_RIGHT_INDX
    #     ;
    # ------------------------------------------------------------

    def enterSPEC_LEVEL(self):
        self.spec_condition = []

    def exitSPEC_LEVEL(self):
        roof_val = 'True' if '@roof' in self.spec_condition else 'False'
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+'roofLevels = '+roof_val )
        self.attribCommaStack[-1] = ",\n"
        all_val = 'True' if 'all' in self.spec_condition else 'False'
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+'allLevels  = '+all_val )
        self.attribCommaStack[-1] = ",\n"
        self.spec_condition = []


    def enterSPEC_ROOF(self,cond):
        self.spec_condition.append(cond)

    def enterSPEC_FULL_INDX(self, index_text):
        indices = index_text.split(':')
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+'indices = ('+indices[0]+','+indices[1]+')' )
        self.attribCommaStack[-1] = ",\n"

    def enterSPEC_SINGLE(self,index_text):
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+'indices = ('+index_text+','+index_text+')' )
        self.attribCommaStack[-1] = ",\n"

    def enterSPEC_LEFT_INDX(self, index_text):
        indices = index_text.split(':')
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+'indices = ('+indices[0]+',-1)' )
        self.attribCommaStack[-1] = ",\n"

    def enterSPEC_RIGHT_INDX(self, index_text):
        indices = index_text.split(':')
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+'indices = (0,'+indices[1]+')' )
        self.attribCommaStack[-1] = ",\n"

    # ------------------------------------------------------------
    # arith_atom
    #     : 'item' '.' IDENTIFIER                                 # ATOM_SINGLE
    #     | 'item' '.' IDENTIFIER '.' IDENTIFIER                  # ATOM_SINGLE
    #     | 'item' '.' IDENTIFIER LBRACK STRING_LITERAL RBRACK    # ATOM_FROMATTR
    #     | 'item' LBRACK STRING_LITERAL RBRACK                   # ATOM_FROMATTR_SHORT
    #|    | 'style' '.' IDENTIFIER                                # ATOM_STYLE
    #     | identifier                                            # ATOM_IDENT
    #     | NUMBER                                                # ATOM_IDENT
    #     | STRING_LITERAL                                        # ATOM_IDENT
    # ------------------------------------------------------------

    def enterATOM_SINGLE(self,atom):
        self.write(atom)

    def enterATOM_FROMATTR(self,ident,literal):
        if self.conditionContext or self.conditionalContext:
            self.write( 'item.' + ident +'.getStyleBlockAttr(' + literal + ')' )
        else:
#            self.write(self.alterCommaStack[-1])
            identifier = ident.capitalize()
            self.write("FromStyleBlockAttr("+literal+",FromStyleBlockAttr."+identifier+")")
 #           self.alterCommaStack[-1] = ",\n"

    def enterATOM_FROMATTR_SHORT(self,literal):
        if self.conditionContext:
            self.write( 'item.getStyleBlockAttr(' + literal + ')' )
        else:
            self.write(self.alterCommaStack[-1])
            self.write(self.indent()+"FromStyleBlockAttr("+literal+")")
            self.alterCommaStack[-1] = ",\n"

    def enterATOM_STYLE(self,identifier):
        if self.conditionContext:
            self.write( 'self.' + identifier )
        else:
            self.write(self.alterCommaStack[-1])
            self.write(self.indent()+'self.' + identifier)
            self.alterCommaStack[-1] = ",\n"

    def enterATOM_IDENT(self,ident):
        self.write(ident)

    def enterConst_atom(self,atom):
        const = self.literalize(atom,True)
        self.write( ',\n'+self.indent()+"Constant(" + const + ')' )

    # ------------------------------------------------------------
    #   ... and all the remaining details
    # ------------------------------------------------------------
    def enterDef_name(self,definition):
        self.write(self.attribCommaStack[-1])
        self.write(self.indent()+'defName = "' + definition + '"' )
        self.attribCommaStack[-1] = ",\n"

    def enterConstant(self,text):
        self.enterCONST(text)

    def enterSimple_expr(self,text):
        if self.smoothContext:
            return
        if text in ('true','false'):
            expr = text.capitalize()
        else:
            expr = self.replaceColorsInText(text)
            expr = self.literalize(expr,False)
        if self.alternativesContext or self.conditionalContext: # ???or self.context  in ( "conditional" ):
            # self.write(self.alterCommaStack[-1])
            self.write( self.indent()+"Constant(" + expr + ')' )
        else:
            self.write(expr)

    def enterAri_lparen(self):
        self.write( ' (' ) 

    def enterAri_rparen(self):
        self.write( ') ' ) 

    def enterIdentifier(self, ident):
        if self.smoothContext and ident in ('smooth','flat','horizontal','side','all') :
            self.write('smoothness.' + ident )

    def enterInop(self,op):
        self.write( ' '+op+' ' )

    def enterRelop(self,op):
        self.write( ' '+op+' ' )

    def enterLogicop(self,op):
        self.write( ' '+op+' ' )

    def enterNotop(self,op):
        self.write( ' '+op+' ' )

    def enterArith_op(self,op):
        self.write( ' '+op+' ' )
