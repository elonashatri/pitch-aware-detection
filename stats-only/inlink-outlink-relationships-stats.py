import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd

# Folder with XML files
annotation_dir = "/homes/es314/omr-objdet-benchmark/data/annotations"

# Lookup: node ID → class name
id_to_class = {}

# Link mappings between class names
inlink_classes = defaultdict(list)   # target_class ← source_class
outlink_classes = defaultdict(list)  # source_class → target_class

# Pass 1: Build ID-to-class mapping
for filename in os.listdir(annotation_dir):
    if not filename.endswith(".xml"):
        continue
    filepath = os.path.join(annotation_dir, filename)
    tree = ET.parse(filepath)
    root = tree.getroot()

    for page in root.findall("Page"):
        for node in page.find("Nodes").findall("Node"):
            node_id_elem = node.find("Id")
            class_name_elem = node.find("ClassName")
            if node_id_elem is not None and class_name_elem is not None:
                node_id = node_id_elem.text.strip()
                class_name = class_name_elem.text.strip()
                id_to_class[node_id] = class_name

# Pass 2: Build class-to-class link mappings
for filename in os.listdir(annotation_dir):
    if not filename.endswith(".xml"):
        continue
    filepath = os.path.join(annotation_dir, filename)
    tree = ET.parse(filepath)
    root = tree.getroot()

    for page in root.findall("Page"):
        for node in page.find("Nodes").findall("Node"):
            node_id_elem = node.find("Id")
            class_name_elem = node.find("ClassName")
            if node_id_elem is None or class_name_elem is None:
                continue
            node_id = node_id_elem.text.strip()
            class_name = class_name_elem.text.strip()

            # Outlinks
            outlinks_elem = node.find("Outlinks")
            if outlinks_elem is not None:
                for target_id in outlinks_elem.text.strip().split():
                    if target_id in id_to_class:
                        target_class = id_to_class[target_id]
                        outlink_classes[class_name].append(target_class)
                        inlink_classes[target_class].append(class_name)

            # Inlinks
            inlinks_elem = node.find("Inlinks")
            if inlinks_elem is not None:
                for source_id in inlinks_elem.text.strip().split():
                    if source_id in id_to_class:
                        source_class = id_to_class[source_id]
                        inlink_classes[class_name].append(source_class)
                        outlink_classes[source_class].append(class_name)

# Aggregate and convert to DataFrames
inlink_df = pd.DataFrame([(target, src) for target, sources in inlink_classes.items() for src in sources],
                         columns=["TargetClass", "SourceClass"])
outlink_df = pd.DataFrame([(src, target) for src, targets in outlink_classes.items() for target in targets],
                          columns=["SourceClass", "TargetClass"])

# Optional: collapse counts
inlink_summary = inlink_df.groupby(["SourceClass", "TargetClass"]).size().reset_index(name="Count")
outlink_summary = outlink_df.groupby(["SourceClass", "TargetClass"]).size().reset_index(name="Count")

# Unique class names that have outgoing connections
classes_with_inlinks = sorted(inlink_classes.keys())

print(f"\n✅ {len(classes_with_inlinks)} unique classes have at least one outlink:")
for cls in classes_with_inlinks:
    print(f" - {cls}")


outlink_summary.to_csv("class_outlinks_summary.csv", index=False)
inlink_summary.to_csv("class_inlinks_summary.csv", index=False)

# # Show results
# print("\n=== Inlink Class Relationships ===")
# print(inlink_summary.sort_values(by="Count", ascending=False).head(100))

# print("\n=== Outlink Class Relationships ===")
# print(outlink_summary.sort_values(by="Count", ascending=False).head(100))
