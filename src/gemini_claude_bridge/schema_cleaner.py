"""
JSON Schema cleaner for Gemini API compatibility.

Gemini API only supports a subset of JSON Schema. This module recursively
strips unsupported fields like additionalProperties, format, oneOf, etc.
"""

from typing import Dict, Any


# Fields that Gemini API does not support
UNSUPPORTED_SCHEMA_KEYS = [
    # Top-level extensions
    "$schema", "additionalProperties",
    # Validation constraints
    "format", "minimum", "maximum", "minLength", "maxLength",
    "minItems", "maxItems", "pattern", "exclusiveMinimum", "exclusiveMaximum",
    # Metadata
    "default", "examples", "const", "title",
    # Composition
    "oneOf", "anyOf", "allOf", "not",
    # Other
    "nullable", "$ref", "$id", "$comment",
    # Object schema extensions
    "propertyNames", "patternProperties", "unevaluatedProperties",
    "dependentSchemas", "dependentRequired", "if", "then", "else",
]


def clean_schema_for_gemini(schema: Dict[str, Any]) -> None:
    """
    Recursively remove JSON Schema fields that Gemini API doesn't support.

    Gemini only supports: type, properties, required, items, enum, description.

    Args:
        schema: JSON Schema dict, modified in-place.
    """
    if not isinstance(schema, dict):
        return

    for key in UNSUPPORTED_SCHEMA_KEYS:
        schema.pop(key, None)

    # Recurse into properties
    if "properties" in schema and isinstance(schema["properties"], dict):
        for prop_schema in schema["properties"].values():
            if isinstance(prop_schema, dict):
                clean_schema_for_gemini(prop_schema)

    # Recurse into items (array type)
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, dict):
            clean_schema_for_gemini(items)
        elif isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    clean_schema_for_gemini(item)
