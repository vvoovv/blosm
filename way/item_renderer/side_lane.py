from ...util.geometry_nodes import setGnInput
from .section import Section
from util.blender import addGeometryNodesModifier
from ..asset_store import AssetType, AssetPart


class SideLane(Section):
    
    def requestNodeGroups(self, nodeGroupNames):
        #super().requestNodeGroups(nodeGroupNames)
        #nodeGroupNames.add("blosm_side_lane_transition")
        return
    
    def setNodeGroups(self, nodeGroups):
        #super().setNodeGroups(nodeGroups)
        #self.gnSideLaneTransition = nodeGroups["blosm_side_lane_transition"]
        return
    
    def finalizeItem(self, section, itemIndex):
        sectionNarrower, sectionWider = (section.pred, section.succ) if section.totalLanesIncreased else (section.succ, section.pred)
        section.width = sectionNarrower.width
        section.offset = sectionNarrower.offset
        
        m = addGeometryNodesModifier(section.street.obj3d, self.gnSideLaneTransition, "Side Lane Transition")
        # Length of the Transition Lane
        setGnInput(m, "Input_2", section.length)
        # Width of the Transition Lane
        setGnInput(m, "Input_3", sectionWider.width - sectionNarrower.width)
        # Width of the Street Section
        setGnInput(m, "Input_4", sectionNarrower.width)
        # Offset of the Street Section
        setGnInput(m, "Input_5", sectionNarrower.offset)
        # Lane on the Right
        setGnInput(m, "Input_6", section.laneR)
        # Material
        self.setMaterial(m, "Input_7", AssetType.material, None, AssetPart.side_lane_transition, section.getStyleBlockAttr("cl"))
        if itemIndex:
            setGnInput(m, "Input_8", itemIndex)
        # Number of Lanes is Increased
        setGnInput(m, "Input_10", section.totalLanesIncreased)
        
        super().finalizeItem(section, itemIndex)
    
    def initItemPolyline1(self, section, singleItem):
        return
    
    def initItemPolyline2(self, section, singleItem):
        return