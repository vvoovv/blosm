import bpy


class GpBase:
    
    def __init__(self):
        # Create a grease pencil object
        gp = bpy.data.grease_pencils.new("Grease Pencil")
        gpo = bpy.data.objects.new(gp.name, gp)
        bpy.context.collection.objects.link(gpo)
        
        # Create a layer
        gpLayer = gp.layers.new("Main")
        
        # Create a frame
        self.gpFrame = gpLayer.frames.new(bpy.context.scene.frame_current)
        
        # Adding a material and setting its properties
        material = bpy.data.materials.new(name="Black")
        bpy.data.materials.create_gpencil_data(material)
        gpo.data.materials.append(material)
        material.grease_pencil.color = (0.0, 0.0, 0.0, 1.0)
    
    def preprocess(self, buildingsP):
        # <buildingsP> means "buildings from the parser"
        return
    
    def cleanup(self):
        return
    
    def do(self, buildingP, style, globalRenderer):
        # building footprint
        if not buildingP.parts or buildingP.alsoPart:
            self.makeStrokes(buildingP.footprint)
        # building parts
        for part in buildingP.parts:
            self.makeStrokes(part.footprint)
    
    def makeStrokes(self, footprint):
        verts = footprint.building.renderInfo.verts
        offsetVertex = footprint.building.renderInfo.offsetVertex
        for facade in footprint.facades:
            vertBL = (verts[facade.indices[0]] + offsetVertex) if offsetVertex else verts[facade.indices[0]]
            vertBR = (verts[facade.indices[1]] + offsetVertex) if offsetVertex else verts[facade.indices[1]]
            vertTR = (verts[facade.indices[2]] + offsetVertex) if offsetVertex else verts[facade.indices[2]]
            self.makeStroke(vertBL, vertBR)
            self.makeStroke(vertBR, vertTR)
    
    def makeStroke(self, vert1, vert2):
        gpStroke = self.gpFrame.strokes.new()
        gpStroke.line_width = 100
        gpStroke.start_cap_mode = 'ROUND'
        gpStroke.end_cap_mode = 'ROUND'
        gpStroke.use_cyclic = False
        
        # Create slots for the given number of points
        gpStroke.points.add(2)
    
        # Fill the slots with the points
        gpStroke.points[0].co = vert1
        gpStroke.points[1].co = vert2