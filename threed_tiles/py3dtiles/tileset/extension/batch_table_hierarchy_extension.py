from __future__ import annotations

from typing import Any

from py3dtiles.tileset.extension.base_extension import BaseExtension
from py3dtiles.typing import ExtensionDictType, HierarchyClassDictType


class HierarchyInstance:
    def __init__(
        self,
        properties: dict[str, Any] | None = None,
        parents: list[int | HierarchyInstance] | None = None,
    ):
        if properties is not None:
            self.properties = properties
        else:
            self.properties = {}

        if parents is not None:
            self.parents = parents
        else:
            self.parents = []


class HierarchyClass:
    def __init__(self, name: str, property_names: list[str]) -> None:
        self.name = name
        self.instances: list[HierarchyInstance] = []
        self.property_names = property_names

    def add_instance(
        self,
        properties: dict[str, Any],
        parents: list[int | HierarchyInstance] | None = None,
    ) -> HierarchyInstance:
        hierarchy_instance = HierarchyInstance(parents=parents)
        for name in self.property_names:
            if name in properties:
                hierarchy_instance.properties[name] = properties[name]
        self.instances.append(hierarchy_instance)
        return hierarchy_instance

    def to_dict(self) -> HierarchyClassDictType:
        return {
            "name": self.name,
            "length": len(self.instances),
            "instances": {
                name: [instance.properties[name] for instance in self.instances]
                for name in self.property_names
            },
        }


class BatchTableHierarchy(BaseExtension):
    """
    Batch Table Hierarchy (BTH) is a BaseExtension of a Batch Table.
    """

    def __init__(self) -> None:
        super().__init__("3DTILES_batch_table_hierarchy")
        self.classes: list[HierarchyClass] = []
        self.instancesLength = 0

    @classmethod
    def from_dict(cls, extension_dict: ExtensionDictType) -> BatchTableHierarchy:
        raise NotImplementedError(
            "The load from a dictionary of the BatchTableHierarchy is not implemented."
        )

    def add_class(self, class_name: str, property_names: list[str]) -> HierarchyClass:
        hierarchy_class = HierarchyClass(class_name, property_names)
        self.classes.append(hierarchy_class)
        return hierarchy_class

    def get_instance_parent_indexes(self, instance: HierarchyInstance) -> list[int]:
        parent_indexes = []
        for parent in instance.parents:
            if isinstance(parent, int):
                parent_indexes.append(parent)
            else:
                index = 0
                for hierarchy_class in self.classes:
                    for hierarchy_instance in hierarchy_class.instances:
                        if parent == hierarchy_instance:
                            parent_indexes.append(index)
                        index += 1
        return parent_indexes

    def to_dict(self) -> ExtensionDictType:
        dict_data: ExtensionDictType = {
            "classes": [],
            "classIds": [],
            "parentCounts": [],
            "parentIds": [],
            "instancesLength": 0,
        }

        for class_id, hierarchy_class in enumerate(self.classes):
            dict_data["classes"].append(hierarchy_class.to_dict())
            for instance in hierarchy_class.instances:
                dict_data["classIds"].append(class_id)
                dict_data["parentCounts"].append(len(instance.parents))
                dict_data["parentIds"].extend(
                    self.get_instance_parent_indexes(instance)
                )
            dict_data["instancesLength"] += len(hierarchy_class.instances)

        return dict_data
