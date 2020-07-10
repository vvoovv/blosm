# Generated from pml.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .pmlParser import pmlParser
else:
    from pmlParser import pmlParser

# This class defines a complete listener for a parse tree produced by pmlParser.
class pmlListener(ParseTreeListener):

    # Enter a parse tree produced by pmlParser#NAMED.
    def enterNAMED(self, ctx:pmlParser.NAMEDContext):
        pass

    # Exit a parse tree produced by pmlParser#NAMED.
    def exitNAMED(self, ctx:pmlParser.NAMEDContext):
        pass


    # Enter a parse tree produced by pmlParser#UNNAMED.
    def enterUNNAMED(self, ctx:pmlParser.UNNAMEDContext):
        pass

    # Exit a parse tree produced by pmlParser#UNNAMED.
    def exitUNNAMED(self, ctx:pmlParser.UNNAMEDContext):
        pass


    # Enter a parse tree produced by pmlParser#named_block.
    def enterNamed_block(self, ctx:pmlParser.Named_blockContext):
        pass

    # Exit a parse tree produced by pmlParser#named_block.
    def exitNamed_block(self, ctx:pmlParser.Named_blockContext):
        pass


    # Enter a parse tree produced by pmlParser#elements.
    def enterElements(self, ctx:pmlParser.ElementsContext):
        pass

    # Exit a parse tree produced by pmlParser#elements.
    def exitElements(self, ctx:pmlParser.ElementsContext):
        pass


    # Enter a parse tree produced by pmlParser#element.
    def enterElement(self, ctx:pmlParser.ElementContext):
        pass

    # Exit a parse tree produced by pmlParser#element.
    def exitElement(self, ctx:pmlParser.ElementContext):
        pass


    # Enter a parse tree produced by pmlParser#attributes.
    def enterAttributes(self, ctx:pmlParser.AttributesContext):
        pass

    # Exit a parse tree produced by pmlParser#attributes.
    def exitAttributes(self, ctx:pmlParser.AttributesContext):
        pass


    # Enter a parse tree produced by pmlParser#attribute.
    def enterAttribute(self, ctx:pmlParser.AttributeContext):
        pass

    # Exit a parse tree produced by pmlParser#attribute.
    def exitAttribute(self, ctx:pmlParser.AttributeContext):
        pass


    # Enter a parse tree produced by pmlParser#sym_expression.
    def enterSym_expression(self, ctx:pmlParser.Sym_expressionContext):
        pass

    # Exit a parse tree produced by pmlParser#sym_expression.
    def exitSym_expression(self, ctx:pmlParser.Sym_expressionContext):
        pass


    # Enter a parse tree produced by pmlParser#use_expression.
    def enterUse_expression(self, ctx:pmlParser.Use_expressionContext):
        pass

    # Exit a parse tree produced by pmlParser#use_expression.
    def exitUse_expression(self, ctx:pmlParser.Use_expressionContext):
        pass


    # Enter a parse tree produced by pmlParser#smooth_expression.
    def enterSmooth_expression(self, ctx:pmlParser.Smooth_expressionContext):
        pass

    # Exit a parse tree produced by pmlParser#smooth_expression.
    def exitSmooth_expression(self, ctx:pmlParser.Smooth_expressionContext):
        pass


    # Enter a parse tree produced by pmlParser#markup_block.
    def enterMarkup_block(self, ctx:pmlParser.Markup_blockContext):
        pass

    # Exit a parse tree produced by pmlParser#markup_block.
    def exitMarkup_block(self, ctx:pmlParser.Markup_blockContext):
        pass


    # Enter a parse tree produced by pmlParser#expression.
    def enterExpression(self, ctx:pmlParser.ExpressionContext):
        pass

    # Exit a parse tree produced by pmlParser#expression.
    def exitExpression(self, ctx:pmlParser.ExpressionContext):
        pass


    # Enter a parse tree produced by pmlParser#alternatives.
    def enterAlternatives(self, ctx:pmlParser.AlternativesContext):
        pass

    # Exit a parse tree produced by pmlParser#alternatives.
    def exitAlternatives(self, ctx:pmlParser.AlternativesContext):
        pass


    # Enter a parse tree produced by pmlParser#ATTR.
    def enterATTR(self, ctx:pmlParser.ATTRContext):
        pass

    # Exit a parse tree produced by pmlParser#ATTR.
    def exitATTR(self, ctx:pmlParser.ATTRContext):
        pass


    # Enter a parse tree produced by pmlParser#BUILDATTR.
    def enterBUILDATTR(self, ctx:pmlParser.BUILDATTRContext):
        pass

    # Exit a parse tree produced by pmlParser#BUILDATTR.
    def exitBUILDATTR(self, ctx:pmlParser.BUILDATTRContext):
        pass


    # Enter a parse tree produced by pmlParser#RANDN.
    def enterRANDN(self, ctx:pmlParser.RANDNContext):
        pass

    # Exit a parse tree produced by pmlParser#RANDN.
    def exitRANDN(self, ctx:pmlParser.RANDNContext):
        pass


    # Enter a parse tree produced by pmlParser#RANDW.
    def enterRANDW(self, ctx:pmlParser.RANDWContext):
        pass

    # Exit a parse tree produced by pmlParser#RANDW.
    def exitRANDW(self, ctx:pmlParser.RANDWContext):
        pass


    # Enter a parse tree produced by pmlParser#COND.
    def enterCOND(self, ctx:pmlParser.CONDContext):
        pass

    # Exit a parse tree produced by pmlParser#COND.
    def exitCOND(self, ctx:pmlParser.CONDContext):
        pass


    # Enter a parse tree produced by pmlParser#USEFROM.
    def enterUSEFROM(self, ctx:pmlParser.USEFROMContext):
        pass

    # Exit a parse tree produced by pmlParser#USEFROM.
    def exitUSEFROM(self, ctx:pmlParser.USEFROMContext):
        pass


    # Enter a parse tree produced by pmlParser#PERBUILD.
    def enterPERBUILD(self, ctx:pmlParser.PERBUILDContext):
        pass

    # Exit a parse tree produced by pmlParser#PERBUILD.
    def exitPERBUILD(self, ctx:pmlParser.PERBUILDContext):
        pass


    # Enter a parse tree produced by pmlParser#RGB.
    def enterRGB(self, ctx:pmlParser.RGBContext):
        pass

    # Exit a parse tree produced by pmlParser#RGB.
    def exitRGB(self, ctx:pmlParser.RGBContext):
        pass


    # Enter a parse tree produced by pmlParser#RGBA.
    def enterRGBA(self, ctx:pmlParser.RGBAContext):
        pass

    # Exit a parse tree produced by pmlParser#RGBA.
    def exitRGBA(self, ctx:pmlParser.RGBAContext):
        pass


    # Enter a parse tree produced by pmlParser#CONST.
    def enterCONST(self, ctx:pmlParser.CONSTContext):
        pass

    # Exit a parse tree produced by pmlParser#CONST.
    def exitCONST(self, ctx:pmlParser.CONSTContext):
        pass


    # Enter a parse tree produced by pmlParser#NESTED.
    def enterNESTED(self, ctx:pmlParser.NESTEDContext):
        pass

    # Exit a parse tree produced by pmlParser#NESTED.
    def exitNESTED(self, ctx:pmlParser.NESTEDContext):
        pass


    # Enter a parse tree produced by pmlParser#ARITH.
    def enterARITH(self, ctx:pmlParser.ARITHContext):
        pass

    # Exit a parse tree produced by pmlParser#ARITH.
    def exitARITH(self, ctx:pmlParser.ARITHContext):
        pass


    # Enter a parse tree produced by pmlParser#nested_list.
    def enterNested_list(self, ctx:pmlParser.Nested_listContext):
        pass

    # Exit a parse tree produced by pmlParser#nested_list.
    def exitNested_list(self, ctx:pmlParser.Nested_listContext):
        pass


    # Enter a parse tree produced by pmlParser#def_name.
    def enterDef_name(self, ctx:pmlParser.Def_nameContext):
        pass

    # Exit a parse tree produced by pmlParser#def_name.
    def exitDef_name(self, ctx:pmlParser.Def_nameContext):
        pass


    # Enter a parse tree produced by pmlParser#conditional.
    def enterConditional(self, ctx:pmlParser.ConditionalContext):
        pass

    # Exit a parse tree produced by pmlParser#conditional.
    def exitConditional(self, ctx:pmlParser.ConditionalContext):
        pass


    # Enter a parse tree produced by pmlParser#condition.
    def enterCondition(self, ctx:pmlParser.ConditionContext):
        pass

    # Exit a parse tree produced by pmlParser#condition.
    def exitCondition(self, ctx:pmlParser.ConditionContext):
        pass


    # Enter a parse tree produced by pmlParser#bool_expr.
    def enterBool_expr(self, ctx:pmlParser.Bool_exprContext):
        pass

    # Exit a parse tree produced by pmlParser#bool_expr.
    def exitBool_expr(self, ctx:pmlParser.Bool_exprContext):
        pass


    # Enter a parse tree produced by pmlParser#cmp_expr.
    def enterCmp_expr(self, ctx:pmlParser.Cmp_exprContext):
        pass

    # Exit a parse tree produced by pmlParser#cmp_expr.
    def exitCmp_expr(self, ctx:pmlParser.Cmp_exprContext):
        pass


    # Enter a parse tree produced by pmlParser#INNESTED.
    def enterINNESTED(self, ctx:pmlParser.INNESTEDContext):
        pass

    # Exit a parse tree produced by pmlParser#INNESTED.
    def exitINNESTED(self, ctx:pmlParser.INNESTEDContext):
        pass


    # Enter a parse tree produced by pmlParser#arith_expr.
    def enterArith_expr(self, ctx:pmlParser.Arith_exprContext):
        pass

    # Exit a parse tree produced by pmlParser#arith_expr.
    def exitArith_expr(self, ctx:pmlParser.Arith_exprContext):
        pass


    # Enter a parse tree produced by pmlParser#ATOM_SINGLE.
    def enterATOM_SINGLE(self, ctx:pmlParser.ATOM_SINGLEContext):
        pass

    # Exit a parse tree produced by pmlParser#ATOM_SINGLE.
    def exitATOM_SINGLE(self, ctx:pmlParser.ATOM_SINGLEContext):
        pass


    # Enter a parse tree produced by pmlParser#ATOM_FROMATTR.
    def enterATOM_FROMATTR(self, ctx:pmlParser.ATOM_FROMATTRContext):
        pass

    # Exit a parse tree produced by pmlParser#ATOM_FROMATTR.
    def exitATOM_FROMATTR(self, ctx:pmlParser.ATOM_FROMATTRContext):
        pass


    # Enter a parse tree produced by pmlParser#ATOM_FROMATTR_SHORT.
    def enterATOM_FROMATTR_SHORT(self, ctx:pmlParser.ATOM_FROMATTR_SHORTContext):
        pass

    # Exit a parse tree produced by pmlParser#ATOM_FROMATTR_SHORT.
    def exitATOM_FROMATTR_SHORT(self, ctx:pmlParser.ATOM_FROMATTR_SHORTContext):
        pass


    # Enter a parse tree produced by pmlParser#ATOM_IDENT.
    def enterATOM_IDENT(self, ctx:pmlParser.ATOM_IDENTContext):
        pass

    # Exit a parse tree produced by pmlParser#ATOM_IDENT.
    def exitATOM_IDENT(self, ctx:pmlParser.ATOM_IDENTContext):
        pass


    # Enter a parse tree produced by pmlParser#ari_lparen.
    def enterAri_lparen(self, ctx:pmlParser.Ari_lparenContext):
        pass

    # Exit a parse tree produced by pmlParser#ari_lparen.
    def exitAri_lparen(self, ctx:pmlParser.Ari_lparenContext):
        pass


    # Enter a parse tree produced by pmlParser#ari_rparen.
    def enterAri_rparen(self, ctx:pmlParser.Ari_rparenContext):
        pass

    # Exit a parse tree produced by pmlParser#ari_rparen.
    def exitAri_rparen(self, ctx:pmlParser.Ari_rparenContext):
        pass


    # Enter a parse tree produced by pmlParser#const_atom.
    def enterConst_atom(self, ctx:pmlParser.Const_atomContext):
        pass

    # Exit a parse tree produced by pmlParser#const_atom.
    def exitConst_atom(self, ctx:pmlParser.Const_atomContext):
        pass


    # Enter a parse tree produced by pmlParser#constant.
    def enterConstant(self, ctx:pmlParser.ConstantContext):
        pass

    # Exit a parse tree produced by pmlParser#constant.
    def exitConstant(self, ctx:pmlParser.ConstantContext):
        pass


    # Enter a parse tree produced by pmlParser#simple_expr.
    def enterSimple_expr(self, ctx:pmlParser.Simple_exprContext):
        pass

    # Exit a parse tree produced by pmlParser#simple_expr.
    def exitSimple_expr(self, ctx:pmlParser.Simple_exprContext):
        pass


    # Enter a parse tree produced by pmlParser#element_name.
    def enterElement_name(self, ctx:pmlParser.Element_nameContext):
        pass

    # Exit a parse tree produced by pmlParser#element_name.
    def exitElement_name(self, ctx:pmlParser.Element_nameContext):
        pass


    # Enter a parse tree produced by pmlParser#attr_name.
    def enterAttr_name(self, ctx:pmlParser.Attr_nameContext):
        pass

    # Exit a parse tree produced by pmlParser#attr_name.
    def exitAttr_name(self, ctx:pmlParser.Attr_nameContext):
        pass


    # Enter a parse tree produced by pmlParser#identifier.
    def enterIdentifier(self, ctx:pmlParser.IdentifierContext):
        pass

    # Exit a parse tree produced by pmlParser#identifier.
    def exitIdentifier(self, ctx:pmlParser.IdentifierContext):
        pass


    # Enter a parse tree produced by pmlParser#relop.
    def enterRelop(self, ctx:pmlParser.RelopContext):
        pass

    # Exit a parse tree produced by pmlParser#relop.
    def exitRelop(self, ctx:pmlParser.RelopContext):
        pass


    # Enter a parse tree produced by pmlParser#logicop.
    def enterLogicop(self, ctx:pmlParser.LogicopContext):
        pass

    # Exit a parse tree produced by pmlParser#logicop.
    def exitLogicop(self, ctx:pmlParser.LogicopContext):
        pass


    # Enter a parse tree produced by pmlParser#notop.
    def enterNotop(self, ctx:pmlParser.NotopContext):
        pass

    # Exit a parse tree produced by pmlParser#notop.
    def exitNotop(self, ctx:pmlParser.NotopContext):
        pass


    # Enter a parse tree produced by pmlParser#inop.
    def enterInop(self, ctx:pmlParser.InopContext):
        pass

    # Exit a parse tree produced by pmlParser#inop.
    def exitInop(self, ctx:pmlParser.InopContext):
        pass


    # Enter a parse tree produced by pmlParser#arith_op.
    def enterArith_op(self, ctx:pmlParser.Arith_opContext):
        pass

    # Exit a parse tree produced by pmlParser#arith_op.
    def exitArith_op(self, ctx:pmlParser.Arith_opContext):
        pass


    # Enter a parse tree produced by pmlParser#number.
    def enterNumber(self, ctx:pmlParser.NumberContext):
        pass

    # Exit a parse tree produced by pmlParser#number.
    def exitNumber(self, ctx:pmlParser.NumberContext):
        pass


    # Enter a parse tree produced by pmlParser#string_literal.
    def enterString_literal(self, ctx:pmlParser.String_literalContext):
        pass

    # Exit a parse tree produced by pmlParser#string_literal.
    def exitString_literal(self, ctx:pmlParser.String_literalContext):
        pass



del pmlParser