import sys
import argparse
import os
import traceback
import runpy #https://www.tutorialspoint.com/locating-and-executing-python-modules-runpy
from datetime import datetime
import subprocess

parser = argparse.ArgumentParser(description="Pot(j)era videos batch processor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-srcdir", "--source", help="directory with video files", default="./examples/testVideoBatch")
parser.add_argument("-o", "--output", help="directory for csv and debug data output", default="results")
parser.add_argument("-lang", "--language", help="ocr language, can be either rs_latin or rs_cyrillic", default="rs_cyrillic")
parser.add_argument("-csv", "--csvFileName", help="name for csv file", default="questions.csv")
parser.add_argument("-d", "--debugData", help="create frame image files for every image processed. note: can use up a lot of data space!", default="True")
args = parser.parse_args()
config = vars(args)

directoryWithVideoFiles = config['source']
directoryOutput = config['output']
csvFileName = config['csvFileName']
createDebugData = config['debugData']
language = config['language']

if not os.path.isdir(directoryWithVideoFiles):
    print('Incorrect directory for videos: \"%s\"' %directoryWithVideoFiles)
    sys.exit(1)

videoFiles = os.listdir(directoryWithVideoFiles)
totalNumberOfFiles = len(videoFiles)
i = 1

print("Files directory set to: %s" %directoryWithVideoFiles)
print("Found %d files to be processed" %totalNumberOfFiles)
print()

for file in videoFiles:
    print()
    print("***********************************************************************************")
    print("Batch processing \"%s\" file (%d of %d)" %(file, i, totalNumberOfFiles))
    # do something
    try:
        # globals ={
        #     "srcdir": directoryWithVideoFiles,
        #     "file": file,
        #     "o": directoryOutput,
        #     "lang": language,
        #     "csv": csvFileName,
        #     "d": createDebugData
        # }

        #subprocess.Popen("python potera-single-video.py -srcdir \"%s\" -file \"%s\" -o \"%s\" -lang %s -csv \"%s\" -d %s" %(directoryWithVideoFiles, file, directoryOutput, language, csvFileName, createDebugData), shell=True)
        os.system("python potera-single-video.py -srcdir \"%s\" -file \"%s\" -o \"%s\" -lang \"%s\" -csv \"%s\" -d %s" %(directoryWithVideoFiles, file, directoryOutput, language, csvFileName, createDebugData))
        # runpy.run_path("./potera-single-video.py -srcdir %s -file %s -o %s -lang %s -csv %s -d %s" %(directoryWithVideoFiles, file, directoryOutput, language, csvFileName, createDebugData))
        # runpy.run_path("./potera-single-video.py")
    except Exception as e:
        print("Error occurred during execution at " + str(datetime.now().date()) + " {}".format(datetime.now().time()))
        print(traceback.format_exc())
        print(e)
        
    i+=1
    print("***********************************************************************************")