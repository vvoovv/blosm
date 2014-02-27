#!/usr/bin/env python3

import math, os

def getNameX(x):
	prefixX = "E" if x>= 0 else "W"
	return "{}{:03d}".format(prefixX, abs(x))

def getFileName(nameX, y, basepath):
	prefixY = "N" if y>= 0 else "S"
	fileName = "{}{:02d}{}.hgt".format(prefixY, abs(y), nameX)
	fileName = os.path.join(basepath, fileName)
	return fileName

class Srtm:
	
	def __init__(self, extent, basepath):
		self.extent = extent
		self.basepath = basepath
		self.minX = int(math.floor(extent.minLon))
		self.maxX = int(math.floor(extent.maxLon))
		self.minY = int(math.floor(extent.minLat))
		self.maxY = int(math.floor(extent.maxLat))
		self.checkFiles()

	def checkFiles(self):
		# check if all needed files are present
		for x in range(self.minX, self.maxX+1):
			nameX = getNameX(x)
			for y in range(self.minY, self.maxY+1):
				fileName = getFileName(nameX, y, self.basepath)
				if not os.path.exists(fileName): raise Exception("Could find SRTM terrain file {}".format(fileName))
				print(fileName)

	def build(self):
		e = extent
		for x in range(self.minX, self.maxX+1):
			nameX = getNameX(x)
			for y in range(self.minY, self.maxY+1):
				fileName = getFileName(nameX, y, self.basepath)
				# open SRTM file
				with open(fileName, "rb") as f:
					TODO