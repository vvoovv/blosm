@name   "mid rise residential zaandam";

        @meta {
            buildingUse : "residential";
            buildingLaf : "modern";
            height : "mid rise";
        }

        footprint {
            height : attr("height");
            minHeight : attr("min_height");
            numLevels : attr("building:levels") | random_weighted( (4, 10), (5, 40), (6, 10) );
            hasNumLevelsAttr : attr("building:levels");
            minLevel : attr("building:min_level") | 0.0;
            topHeight : 0.;
            levelHeight : random_normal(3.);
            groundLevelHeight : random_normal(4.2);
            bottomHeight : random_normal(1.);
            roofShape : attr("roof:shape") | "flat" | "saltbox" | random_weighted( ("gabled", 10), ("flat", 40) );
            roofHeight : attr("roof:height") | 5.0;
            roofAngle : attr("roof:angle");
            roofDirection : attr("roof:direction") | attr("roof:slope:direction");
            roofOrientation : "across";
            lastLevelOffsetFactor : random_weighted (
                (0., 50), (0.05, 3), (0.1, 5), (0.15, 5), (0.2, 5), (0.25, 5), (0.3, 5),
                (0.35, 5), (0.4, 5), (0.45, 5), (0.5, 3), (0.55, 2), (0.6, 2)
            );
            claddingColor : per_building( random_weighted (  
                ((0.647, 0.165, 0.165, 1.), 1), // brown
                ((0.565, 0.933, 0.565, 1.), 1), // lightgreen
                ((1., 0.855, 0.725, 1.), 1)     // peachpuff
            ));
            claddingMaterial : "brick";
        }

        facade@brown_brick {        
        }

        level@level_window_balcony {
            markup : [
                window {
                    width : 1.8;
                    height : 2.1;
                    rows : 1;
                    panels : 2;
                }
                balcony{}
            ]
        }

        level@staircase {
            // offset : (0.5, units.Level)  // data type not yet implemented
        }

        window@back_facade_window {
            width : 1.2;
            height : 1.8;
            panels : 1;
        }

        window@roof_window {
            width : 0.8;
            height : 0.8;
            rows : 1;
            panels : 1;

        }

        div@window_and_balcony {
            label : "Window and balcony";
            markup : [
                level {
                    use : level_window_balcony; 
                    claddingMaterial : "plaster";
                    indices : (4, -1);
                    claddingColor : (0., 0., 1., 1.);  // blue
                }
                level {
                    use : level_window_balcony;
                    indices : (3, 3);
                    claddingColor : (0., 0.502, 0., 1.); // green
                }
                level{
                    use : level_window_balcony;
                    indices : (0, 2);
                }
                bottom {
                    markup : [
                        window{
                            width : 1.;
                            height : 1.;
                            rows : 1;
                            panels : 1;
                        }
                    ]
                }
            ]
        }

        div@staircase {
            label : "Staircase";
            bottomHeight : 0;
            markup : [
                level {
                    repeat : false;
                    indices : (1, -1);
                    markup : [
                        window {
                            width : 0.8;
                            height : 0.8;
                            rows : 1;
                            panels : 1;
                        }
                    ]
                }
                level {
                    indices : (0, 0);
                    markup : [
                        door {
                            label : "entrance door";
                        }
                    ]
                }
            ]
        }

        div@roof_side {
            width : use_from(main_section);
            symmetry : right-most-of-last;
        }

        facade[item.footprint.height - item.footprint.minHeight < minHeightForLevels] {
            label : "cladding only for too low structures";
        }

        facade[item.front] {
            use: brown_brick;
            label : "front facade";
            symmetry : middle-of-last;
            symmetryFlip : true;
            markup : [
                div {
                    use : window_and_balcony;
                    id : "main_section";
                    label : "Window and Balcony";
                }
                div {
                    use : staircase;
                    label : "Staircase";
                }
            ] 
        }

        facade[item.back] {
            use: brown_brick;
            label : "back facade";
            markup : [
                level{
                    indices : (0, -1);
                    markup : [
                        balcony{}
                        window{
                            use : back_facade_window;
                        }
                        Window{
                            use : back_facade_window;
                        }
                    ]
                }
            ]
        }

        roof {
            claddingMaterial : "brick";
            claddingColor : (0.98, 0.502, 0.447, 1.); // salmon
            faces : smooth;      
            sharpEdges : side; 
        }

        roof-side[item.front] { 
            markup : [
                div {
                    use : roof_side;
                    markup : [
                        // openable skylight or roof window
                        window{
                            use : roof_window;
                        }
                        window{
                            use : roof_window;
                        }
                    ]
                }
            ]
        }

        roof-side[item.back] {
            markup : [
                div {
                    use : roof_side;
                    markup : [
                        dormer{} dormer{}
                    ]
                }
            ]
        }

        ridge {
            markup : [
                div {
                    width : use_from(main_section);
                    repeat : false; 
                    markup : [
                        chimney{}
                    ]
                }
            ]    
        }


