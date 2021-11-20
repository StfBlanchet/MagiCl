# ! /usr/bin/env python3
# coding: utf-8

"""
Regular expressions
"""


EMAIL = r"\w+@\S+\.\w+\.*\w*"
PHONE = r"\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}"
URL = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.\w+\.\w+\.*\w*"

HASHTAG = r"#.*?(?=\s|$)"
MENTION = r"@.*?(?=\s|$)"

CAPS = r"([A-Z]{2,}\s)"
QUEST = r"\?+"
EXCL = r"\!+"

CLEANER = "({}|{}|{})".format(URL, EMAIL, PHONE)
