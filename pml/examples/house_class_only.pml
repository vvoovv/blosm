@name "single family house";

@meta{
  buildingUse: single_family;
  buildingLaf: modern;
  height: "low rise";
}

footprint{
  roofShape: gabled;
  numLevels: 1;
  numRoofLevels: 1;
  markup: [
    facade[item.front] {
      class: facade_front;
    }
    facade[item.back] {
      class: facade_back;
    }
    facade {
      class: facade_side;
    }
  ]
}
