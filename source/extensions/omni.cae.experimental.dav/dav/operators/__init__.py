"""
Operators module for dav package.

This module contains various operators for data analysis and visualization,
including advection, bounds calculation, cell operations, and streamlines.

All operators follow a consistent pattern:
- Operators that take field input accept a field name (str) and retrieve the field from the dataset
- Operators always return a dataset (or datasets) with computed fields added
- Operators that add fields take an optional output field name parameter
"""
