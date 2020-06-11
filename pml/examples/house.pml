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
  random: [
    fooprint{
      use: house_type_1;
      f1CladdingMaterial: brick;
      f1CladdingColor: gray;
      f0CladdingMaterial: brick;
      f0CladdingColor: gray;
    }
    fooprint{
      use: house_type_2;
      f1CladdingMaterial: brick;
      f1CladdingColor: pink;
      f0CladdingMaterial: brick;
      f0CladdingColor: pink;
    }
    fooprint{
      use: house_type_1;
      f1CladdingMaterial: vinyl;
      f1CladdingColor: khaki;
      f0CladdingMaterial: brick;
      f0CladdingColor: brown;
    }
    fooprint{
      use: house_type_2;
      f1CladdingMaterial: vinyl;
      f1CladdingColor: lightblue;
      f0CladdingMaterial: brick;
      f0CladdingColor: darkgold;
    }
  ]
}

footprint@house_type_1{
  markup: [
    facade[item.front] {
      use: facade_front;
    }
    facade[item.back] {
      use: facade_back;
    }
    facade{
      alternate: [
        facade {
          use: facade_side_1;
        }
        facade {
          use: facade_side_2;
        }
      ]
    }
  ]
}

footprint@house_type_2{
  markup: [
    facade[item.front] {
      use: facade_front;
    }
    facade[item.back] {
      use: facade_back;
    }
    facade{
      alternate: [
        facade {
          use: facade_side_1;
        }
        facade {
        // empty facade
        }
      ]
    }
  ]
}

level@ground_floor {
  indices: (0,0);
  claddingMaterial: item.footprint["f0CladdingMaterial"];
  claddingColor: item.footprint["f0CladdingColor"];
}

level@ground_floor {
  indices: (1,1);
  claddingMaterial: item.footprint["f0CladdingMaterial"];
  claddingColor: item.footprint["f0CladdingColor"];
}

facade@facade_front{
  class: facade_front;
  markup: [
    level{
      use: floor_1;
      markup: [
        window{}
      ]
    }
    level{
      use: ground_floor;
      markup: [
        window{} door{} window{}
      ]
    }
  ]
}

facade@facade_back{
  class: facade_back;
  markup: [
    level{
      use: floor_1;
    }
    level{
      use: ground_floor;
      markup: [
        window{} window{} window{}
      ]
    }
  ]
}

facade@facade_side_1{
  class: facade_side_1;
  markup: [
    level{
      use: floor_1;
      markup: [
        window{} window{} window{}
      ]
    }
    level{
      use: ground_floor;
      markup: [
        window{} window{} window{}
      ]
    }
  ]
}

facade@facade_side_2{
  class: facade_side_2;
  markup: [
    level{
      use: floor_1;
      markup: [
        window{} balcony{} window{}
      ]
    }
    level{
      use: ground_floor;
      markup: [
        window{} window{}
      ]
    }
  ]
}
