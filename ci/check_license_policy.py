#!/usr/bin/env python3
"""Validate the repository's source-license policy inputs."""

from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
for name in ("LICENSE", "COPYRIGHT", "OAT.xml"):
    if not (root / name).is_file():
        raise SystemExit(f"missing license policy file: {name}")
tree = ET.parse(root / "OAT.xml")
licenses = {item.attrib.get("name") for item in tree.findall(".//policyitem[@type='license']")}
if "Apache-2.0" not in licenses:
    raise SystemExit("OAT.xml does not allow the repository Apache-2.0 license")
print("Validated LICENSE, COPYRIGHT, and OAT.xml policy")
