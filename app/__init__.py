#!/usr/bin/env python
# coding: utf8

"""
Flask init and config
"""

from os import urandom

from flask import Flask


app = Flask(__name__)
app.config["SECRET_KEY"] = urandom(32)

from app import api
