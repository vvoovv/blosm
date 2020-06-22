@name "single family house";

@meta{
  buildingUse: single_family;
}

footprint{
  roofShape: gabled;
  numLevels: 1;
  numRoofLevels: 1;
  levelHeight: random_normal(3.);
}

facade[item.front] {
  class: facade_front;
}
    
facade[item.back] {
  class: facade_back;
}
    
facade {
  class: facade_side;
}

roof{
  class: roof;
  roofCladdingMaterial: metal;
  roofCladdingColor: darkgray;
}


@name "commercial";

@meta{
  buildingUse: office;
}

footprint{
  roofShape: flat;
  numLevels: attr("building:levels") | random_weighted( (2,2), (3,2), (4,3), (5,1) );
  levelHeight: random_normal(3.);
}

//
// 2 floors
//

facade[item.front and item.footprint["numLevels"]==2] {
  class: facade_front_2floors;
}

facade[item.footprint["numLevels"]==2] {
  class: facade_side_2floors;
}


//
// 3 floors
//

facade[item.front and item.footprint["numLevels"]==3] {
  class: facade_front_3floors;
}

facade[item.footprint["numLevels"]==3] {
  class: facade_side_3floors;
}


//
// 4 floors
//

facade[item.front and item.footprint["numLevels"]==4] {
  class: facade_front_4floors;
}

facade[item.footprint["numLevels"]==4] {
  class: facade_side_4floors;
}


//
// 5 floors
//

facade[item.front and item.footprint["numLevels"]==5] {
  class: facade_front_5floors;
}

facade[item.footprint["numLevels"]==5] {
  class: facade_side_5floors;
}


roof{
  class: roof;
  roofCladdingMaterial: concrete;
  roofCladdingColor: gray;
}