"""
Image format conversion utilities for Gemini ↔ Claude.

Handles conversion between:
- Claude base64 image format
- Claude URL image format
- Gemini inlineData format
- Gemini fileData format
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple

from .constants import Constants

logger = logging.getLogger(__name__)


def convert_image_source(source: Any) -> Optional[Dict[str, Any]]:
    """
    Convert Claude/Gemini image source to Gemini inlineData/fileData format.

    Supported input formats:
    1. Claude base64: {"type": "base64", "media_type": "image/png", "data": "..."}
    2. Claude URL: {"type": "url", "url": "https://..."}
    3. Gemini inlineData: {"inlineData": {"mimeType": "image/png", "data": "..."}}

    Args:
        source: Image source dict.

    Returns:
        Gemini format dict, or None if conversion fails.
    """
    if not source or not isinstance(source, dict):
        return None

    # Gemini inlineData format - passthrough
    if "inlineData" in source:
        inline_data = source["inlineData"]
        if isinstance(inline_data, dict) and "data" in inline_data:
            return source

    source_type = source.get("type")
    media_type = source.get("media_type", "image/jpeg")

    if source_type == "base64":
        data = source.get("data", "")
        if data:
            return {
                "inlineData": {
                    "mimeType": media_type,
                    "data": data,
                }
            }

    elif source_type == "url":
        url = source.get("url", "")
        if url:
            return {
                "fileData": {
                    "mimeType": media_type,
                    "fileUri": url,
                }
            }

    logger.warning(f"Unknown image source type: {source_type}")
    return None


def looks_like_image_data(data: Dict[str, Any]) -> bool:
    """
    Heuristic check whether a dict contains image data.

    Args:
        data: Dict to check.

    Returns:
        True if it looks like image data.
    """
    if not isinstance(data, dict):
        return False

    if data.get("type") == "base64" and "data" in data:
        return True

    if "media_type" in data and "data" in data:
        media_type = data.get("media_type", "")
        if isinstance(media_type, str) and media_type.startswith("image/"):
            return True

    if "inlineData" in data:
        inline_data = data["inlineData"]
        if isinstance(inline_data, dict):
            mime_type = inline_data.get("mimeType", "")
            if isinstance(mime_type, str) and mime_type.startswith("image/") and "data" in inline_data:
                return True

    return False


def parse_tool_result_content(content: Any) -> str:
    """
    Parse tool_result content into plain text.

    Args:
        content: tool_result content (string, list, or dict).

    Returns:
        Plain text output.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == Constants.CONTENT_TEXT:
                    parts.append(item.get("text", ""))
                else:
                    try:
                        parts.append(json.dumps(item, ensure_ascii=False))
                    except (TypeError, ValueError):
                        parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)

    return str(content)


def parse_tool_result_content_with_images(content: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse tool_result content, separating text from images.

    When tool_result contains images (e.g. from Read tool reading a PNG),
    extract them as Gemini inlineData instead of serializing as text.

    Args:
        content: tool_result content.

    Returns:
        (text_output, images) where images are Gemini inlineData dicts.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []

    text_parts: List[str] = []
    image_parts: List[Dict[str, Any]] = []

    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")

                if item_type == Constants.CONTENT_TEXT:
                    text_parts.append(item.get("text", ""))

                elif item_type == Constants.CONTENT_IMAGE:
                    source = item.get("source")
                    if source:
                        image_item = convert_image_source(source)
                        if image_item:
                            image_parts.append(image_item)
                        else:
                            text_parts.append("[Image: conversion failed]")
                    else:
                        text_parts.append("[Image: no source]")

                else:
                    if looks_like_image_data(item):
                        image_item = convert_image_source(item)
                        if image_item:
                            image_parts.append(image_item)
                            continue

                    try:
                        json_str = json.dumps(item, ensure_ascii=False)
                        if len(json_str) > 10000:
                            text_parts.append(f"[Large data truncated: {len(json_str)} chars]")
                        else:
                            text_parts.append(json_str)
                    except (TypeError, ValueError):
                        text_parts.append(str(item))
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts), image_parts

    if isinstance(content, dict):
        content_type = content.get("type")

        if content_type == Constants.CONTENT_TEXT:
            return content.get("text", ""), []

        elif content_type == Constants.CONTENT_IMAGE:
            source = content.get("source")
            if source:
                image_item = convert_image_source(source)
                if image_item:
                    return "", [image_item]
            return "[Image: conversion failed]", []

        else:
            if looks_like_image_data(content):
                image_item = convert_image_source(content)
                if image_item:
                    return "", [image_item]

            try:
                json_str = json.dumps(content, ensure_ascii=False)
                if len(json_str) > 10000:
                    return f"[Large data truncated: {len(json_str)} chars]", []
                return json_str, []
            except (TypeError, ValueError):
                return str(content), []

    return str(content), []
