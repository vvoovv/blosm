@name "high rise";

@meta {
	buildingUse: office;
	buildingLaf: modern;
	height: "high rise";
}

footprint {
	height: attr("height");
	minHeight: attr("min_height");
	hasNumLevelsAttr: attr("building:levels");
	numLevels: attr("building:levels") | random_weighted( (4, 10), (5, 40), (6, 10) );
	minLevel: attr("building:min_level") | 0;
	topHeight: 0.;
	levelHeight: random_normal(3.);
	roofShape: attr("roof:shape") | flat;
	roofHeight: attr("roof:height");
	roofOrientation: attr("roof:orientation");
	claddingMaterial: per_building(
		attr("building:material") | random_weighted( (brick, 1), (plaster, 1) )
	);
	claddingColor: per_building(
		attr("building:colour")
		|
		if (item["claddingMaterial"] == "brick") random_weighted(
			(brown, 1),
			(salmon, 1),
			(maroon, 1)
		)
		|
		if (item["claddingMaterial"] == "plaster") random_weighted(
            (lightsalmon, 1),
            (lightgreen, 1),
            (peachpuff, 1)
		)
		|
		if (item["claddingMaterial"] == "glass") random_weighted(
            (#4e7292, 1),
            (#2b515c, 1),
            (#182e45, 1)
		)
	);
}

facade[
	not item.footprint.numLevels or
	item.footprint.height - item.footprint.minHeight < 1.5 or // minHeightForLevels
	item.width < 1. // minWidthForOpenings
	or item.footprint["roofShape"] in ("gabled", "round", "gambrel", "saltbox")
] {
	label: "cladding only for structures without levels or too low structures or too narrow facades";
}

facade[item.footprint["claddingMaterial"] == "glass"] {
	markup: [
		curtain_wall{}
	]
}

facade {
	markup: [
		level{}
	]
}

roof {
	roofCladdingMaterial:
		attr("roof:material")
		|
		if (item.footprint["roofShape"] == "flat") concrete
		|
		metal
	;
	roofCladdingColor:
		attr("roof:colour")
		|
		if (item["roofCladdingMaterial"] == "concrete") random_weighted(
            (#afafaf, 1),
            (#b2b2a6, 1),
            (#c8c2b6, 1)
		)
		|
		// roofCladdingMaterial == "metal"
		random_weighted(
            (#afafaf, 1),
            (#b2b2a6, 1),
            (#c8c2b6, 1)
		)
	;
	faces: if (item.footprint["roofShape"] in ("dome", "onion")) smooth;
}


@name "residential";

@meta {
	buildingUse: appartments;
	buildingLaf: modern;
	height: "high rise";
}

footprint {
	height: attr("height");
	minHeight: attr("min_height");
	hasNumLevelsAttr: attr("building:levels");
	numLevels: attr("building:levels") | random_weighted( (4, 10), (5, 40), (6, 10) );
	minLevel: attr("building:min_level") | 0;
	topHeight: 0.;
	levelHeight: random_normal(3.);
	roofShape: attr("roof:shape") | flat;
	roofHeight: attr("roof:height");
	roofOrientation: attr("roof:orientation");
	claddingMaterial: per_building(
		attr("building:material") | random_weighted( (brick, 1), (plaster, 1) )
	);
	claddingColor: per_building(
		attr("building:colour")
		|
		if (item["claddingMaterial"] == "brick") random_weighted(
			(brown, 1),
			(salmon, 1),
			(maroon, 1)
		)
		|
		// plaster
		random_weighted(
            (lightsalmon, 1),
            (lightgreen, 1),
            (peachpuff, 1)
		)
	);
}

facade[
	not item.footprint.numLevels or
	item.footprint.height - item.footprint.minHeight < 1.5 or // minHeightForLevels
	item.width < 1. // minWidthForOpenings
	or item.footprint["roofShape"] in ("gabled", "round", "gambrel", "saltbox")
] {
	label: "cladding only for structures without levels or too low structures or too narrow facades";
}

facade {
	markup: [
		level{}
	]
}

roof {
	roofCladdingMaterial:
		attr("roof:material")
		|
		if (item.footprint["roofShape"] == "flat") concrete
		|
		metal
	;
	roofCladdingColor:
		attr("roof:colour")
		|
		if (item["roofCladdingMaterial"] == "concrete") random_weighted(
            (#afafaf, 1),
            (#b2b2a6, 1),
            (#c8c2b6, 1)
		)
		|
		// roofCladdingMaterial == "metal"
		random_weighted(
            (#afafaf, 1),
            (#b2b2a6, 1),
            (#c8c2b6, 1)
		)
	;
	faces: if (item.footprint["roofShape"] in ("dome", "onion")) smooth;
}


@name "place of worship";

footprint {
	height: attr("height");
	minHeight: attr("min_height");
	numLevels: 0;
	topHeight: 0.;
	roofShape: attr("roof:shape") | flat;
	roofHeight: attr("roof:height");
	roofOrientation: attr("roof:orientation");
	claddingMaterial: per_building(
		attr("building:material") | plaster
	);
	claddingColor: per_building(
		attr("building:colour")
		|
		// plaster
		random_weighted(
            (lightsalmon, 1),
            (lightgreen, 1),
            (peachpuff, 1)
		)
	);
}

facade {
// cladding material only
}

roof {
	roofCladdingMaterial: attr("roof:material") | metal;
	roofCladdingColor:
		attr("roof:colour")
		|
		// roofCladdingMaterial == "metal"
		random_weighted(
            (#afafaf, 1),
            (#b2b2a6, 1),
            (#c8c2b6, 1)
		)
	;
	faces: if (item.footprint["roofShape"] in ("dome", "onion")) smooth;
}


@name "man made";

footprint {
	height: attr("height");
	minHeight: attr("min_height");
	numLevels: 0;
	topHeight: 0.;
	roofShape: attr("roof:shape") | flat;
	roofHeight: attr("roof:height");
	roofOrientation: attr("roof:orientation");
	claddingMaterial: per_building(
		attr("building:material") | brick
	);
	claddingColor: per_building(
		attr("building:colour")
		|
		// plaster
		random_weighted(
			(brown, 1),
			(salmon, 1),
			(maroon, 1)
		)
	);
}

facade {
// cladding material only
}

roof {
	roofCladdingMaterial: attr("roof:material") | metal;
	roofCladdingColor:
		attr("roof:colour")
		|
		// roofCladdingMaterial == "metal"
		random_weighted(
            (#afafaf, 1),
            (#b2b2a6, 1),
            (#c8c2b6, 1)
		)
	;
	faces: if (item.footprint["roofShape"] in ("dome", "onion")) smooth;
}