#!/usr/bin/env python
# coding: utf8

"""
Flask config
"""

from os import path
from dotenv import load_dotenv


basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, ".env"))
