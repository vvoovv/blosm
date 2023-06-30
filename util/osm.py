"""
This file is a part of Blosm addon for Blender.
Copyright (C) 2014-2018 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

def assignTags(obj, tags):
    for key in tags:
        obj[key] = tags[key]


def parseNumber(s, defaultValue=None):
    s = s.rstrip()
    if s[-1] == 'm':
        s = s[:-1]
    try:
        n = float(s)
    except ValueError:
        n = defaultValue
    return n
