"""Small text utilities with no model or network dependencies."""
from __future__ import annotations

import re

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}


def decode_delimiter(delimiter: str) -> str:
    """Translate \\n, \\t, \\r and \\\\ while leaving every other character intact.

    Python's unicode_escape codec would also do this, but it corrupts non-ASCII
    delimiters such as the Chinese comma, so only these escapes are handled.
    """
    out: list[str] = []
    chars = iter(delimiter)
    for char in chars:
        if char != "\\":
            out.append(char)
            continue
        following = next(chars, None)
        if following is None:
            out.append("\\")
        elif following in _ESCAPES:
            out.append(_ESCAPES[following])
        else:
            out.extend(("\\", following))
    return "".join(out)


def split_text(text: str, delimiter: str, start_index: int = 0, skip_every: int = 0,
               max_count: int = 10, trim: bool = False, regex: bool = False) -> list[str]:
    if delimiter == "":
        parts = [text.strip()]
    else:
        if regex:
            try:
                pattern = re.compile(delimiter)
            except re.error as error:
                raise ValueError(f"Invalid regular expression delimiter: {error}") from None
            # Cut between matches instead of re.split, so capturing groups never
            # turn separators into output items.
            pieces, start = [], 0
            for match in pattern.finditer(text):
                if match.end() > match.start():
                    pieces.append(text[start:match.start()])
                    start = match.end()
            pieces.append(text[start:])
        else:
            pieces = text.split(decode_delimiter(delimiter))
        parts = [part for part in pieces if part.strip()]
        if trim:
            parts = [part.strip() for part in parts]
    step = skip_every + 1
    return parts[start_index:start_index + max_count * step:step]


class TankTextSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                     "tooltip": "Text to split."}),
                "delimiter": ("STRING", {"default": ",", "dynamicPrompts": False,
                                         "tooltip": r"Separator. Use \n for a new line and \t for a tab. Leave empty to output the whole text as one item."}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                        "tooltip": "Index of the first item to output, counting from 0."}),
                "skip_every": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                                       "tooltip": "Items skipped after each output item. 1 outputs every other item."}),
                "max_count": ("INT", {"default": 10, "min": 1, "max": 100000, "step": 1,
                                      "tooltip": "Maximum number of items to output."}),
                "trim": ("BOOLEAN", {"default": False, "advanced": True,
                                     "tooltip": "Remove spaces and line breaks around each item."}),
                "regex": ("BOOLEAN", {"default": False, "advanced": True,
                                      "tooltip": "Treat the delimiter as a regular expression, for example [,，] for both commas."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("items", "count")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = ("Each item runs connected nodes once.", "Number of items output.")
    FUNCTION = "split"
    CATEGORY = "TankNodes/Text"
    DESCRIPTION = "Split text into a list. Blank items are dropped; connected nodes run once per item."

    def split(self, text: str, delimiter: str, start_index: int, skip_every: int, max_count: int,
              trim: bool = False, regex: bool = False):
        items = split_text(text, delimiter, start_index, skip_every, max_count, trim, regex)
        if not items:
            raise ValueError("Text split produced no items: check the text, delimiter and start index.")
        return (items, len(items))
