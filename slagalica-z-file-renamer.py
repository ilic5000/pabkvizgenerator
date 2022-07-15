import sys
import os
from datetime import datetime

directoryWithVideoFiles = 'D:\Slagalica720p\Slagalica-720p-novi-intro-stara-grafika'

if not os.path.isdir(directoryWithVideoFiles):
    print('Incorrect directory for videos: \"%s\"' %directoryWithVideoFiles)
    sys.exit(1)

videoFiles = sorted(os.listdir(directoryWithVideoFiles))

for oldFileName in videoFiles:

    # attempt 1 
    # beforeDate = oldFileName.rsplit(' ',2)[0]
    # afterDate = oldFileName.rsplit(' ',)[2]
    # date = oldFileName.split(' ')[1]
    # day = date.split('.')[0]
    # month = date.split('.')[1]
    # year = date.split('.')[2]
    # newFileName = "%s.%s.%s %s %s" %(year, month, day, beforeDate, afterDate)
    # print("%s ->" %oldFileName)
    # print(newFileName)
    # print()
    # os.rename("%s\%s" %(directoryWithVideoFiles, oldFileName), "%s\%s" %(directoryWithVideoFiles, newFileName))

    # attempt 2
    # date = oldFileName.split(' ')[0]
    # slagalicaName = oldFileName.split(' ')[1]
    # afterDuplicateDate = oldFileName.split(' ', 3)[3]
    # newFileName = "%s %s %s" %(date, slagalicaName, afterDuplicateDate)

    #attempt 3
    findAndDelete = " 720p_25fps_H264-192kbit_AAC)"
    if oldFileName.find(findAndDelete) >0:
        newFileName = oldFileName.replace(findAndDelete, "").strip()

        print("%s ->" %oldFileName)
        print(newFileName)
        print()
        os.rename("%s\%s" %(directoryWithVideoFiles, oldFileName), "%s\%s" %(directoryWithVideoFiles, newFileName))