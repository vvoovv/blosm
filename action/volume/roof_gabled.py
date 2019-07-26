from .roof_profile import RoofProfile


# Use https://raw.githubusercontent.com/wiki/vvoovv/blender-osm/assets/roof_profiles.blend
# to generate values for a specific profile
_gabledRoof = (
    (
        (0., 0.),
        (0.5, 1.),
        (1., 0.)
    ),
    {
        "numSamples": 10,
        "angleToHeight": 0.5
    }
)


class RoofGabled(RoofProfile):
    
    # default roof height
    height = 4.
    
    def __init__(self):
        super().__init__(_gabledRoof)  