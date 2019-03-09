#! /Users/dpbrinegar/anaconda3/envs/gitHub/bin/python

import re
import sqlite3
import numpy as np
import sys

def post_row(conn, tablename, rec):
    question_marks = ','.join(list('?'*len(rec)))
    command = 'INSERT INTO cnn_results VALUES (%s,%s,%s,%d,%f,%d,%d,%s,%d,%d,%d,%s,%f,%s,%s,%s,%0.16f,%d,%s,%0.16f,%d,%s,%0.16f,%d,%s,%0.16f,%d)' % ("'%s'" % rec[0],
                                                                      "'%s'" % rec[1],
                                                                      "'%s'" % rec[2],
                                                                      int(rec[3]),
                                                                      float(rec[4]),
                                                                      int(rec[5]),
                                                                      int(rec[6]),
                                                                      "'%s'" % rec[7],
                                                                      int(rec[8]),
                                                                      int(rec[9]=='True'),
                                                                      int(rec[10]),
                                                                      "'%s'" % rec[11],
                                                                      float(rec[12]),
                                                                      "'%s'" % rec[13],
                                                                      "'%s'" % rec[14],
                                                                      "'%s'" % rec[15],
                                                                      float(rec[16]),
                                                                      int(rec[17]),
                                                                      "'%s'" % rec[18],
                                                                      float(rec[19]),
                                                                      int(rec[20]),
                                                                      "'%s'" % rec[21],
                                                                      float(rec[22]),
                                                                      int(rec[23]),
                                                                      "'%s'" % rec[24],
                                                                      float(rec[25]),
                                                                      int(rec[26])
                                                                      )
    conn.execute(command)




_db = sqlite3.connect('./NeuralNet/results/cnnGridSearchResults/cnnResults.db')
_cursor = _db.cursor()
_command = 'CREATE TABLE IF NOT EXISTS cnn_results ' + \
                '([id] STRING PRIMARY KEY NOT NULL,' + \
                '[dbfile] STRING,' + \
                '[glovefile] STRING,' + \
                '[vocabsize] INTEGER,' + \
                '[capturefraction] REAL,' + \
                '[convolutionfilters] INTEGER,' + \
                '[convolutionkernel] INTEGER,' + \
                '[convolutionactivation] STRING,' + \
                '[poolsize] INTEGER,' + \
                '[flattenlayers] INTEGER,' + \
                '[denseunits] INTEGER,' + \
                '[denseactivation] STRING,' + \
                '[dropoutfraction] REAL,' + \
                '[outputactivation] STRING,' + \
                '[lossfunction] STRING,' + \
                '[trainaccuracy] STRING,' + \
                '[trainmaxaccuracy] REAL,' + \
                '[trainmaxaccuracyepoch] INTEGER,' + \
                '[valaccuracy] STRING,' + \
                '[valmaxaccuracy] REAL,' + \
                '[valmaxaccuracyepoch] INTEGER,' + \
                '[trainloss] STRING,' + \
                '[trainminloss] REAL,' + \
                '[trainminlossepoch] INTEGER,' + \
                '[valloss] STRING,' + \
                '[valminloss] REAL,' + \
                '[valminlossepoch] INTEGER);'
_cursor.execute(_command)

with open(sys.argv[1]) as _theFile:
    _pandasFlag = True
    _trainAccFlag = False
    _valAccFlag = False
    _trainLossFlag = False
    _valLossFlag = False
    _parms = ['junk', 'id', 'dbfile', 'glovefile', 'vocabsize', 'capturefraction', 'convolutionfilters',
              'convolutionkernel', 'convolutionactivation', 'poolsize', 'flattenlayers', 'denseunits',
              'denseactivation', 'dropoutfraction', 'outputactivation', 'lossfunction']
    _theDict = {}
    j = 0
    _maxid = sys.maxsize

    _row = _theFile.readline()
    while _row:
        if (_pandasFlag) and (_row.find('Pandas') != -1):
            _row = _row.strip().translate(str.maketrans({"'":None, ')':None}))
            _equals = [x.start() for x in re.finditer('=', _row)]
            _commas = [x.start() for x in re.finditer(',', _row)]
            _commas.append(len(_row)+1)
            _parmlist = []
            for k in range(len(_parms)):
                _parmlist.append(_row[_equals[k]+1:_commas[k]])
            _theDict[int(_parmlist[1])] = _parmlist[1:]
            j += 1

        if (_trainAccFlag) and (_row.find('Training Accuracies:') != -1):
            _row = _theFile.readline()
            while len(_row) > 3:
                _row = _row.strip().translate(str.maketrans({',':None, '[':None, ']':None}))
                junk = _row.split()
                junk = [int(junk[0]), [float(x) for x in junk[1:]]]
                _parmlist = _theDict[junk[0]]
                _theDict[junk[0]] = _parmlist + junk[1:] + [np.max(junk[1:])] + [np.argmax(junk[1:])]
                _row = _theFile.readline()
            if len(_row) <= 3:
                _trainAccFlag = False
                _valAccFlag = True

        if (_valAccFlag) and (_row.find('Validation Accuracies:') != -1):
            _row = _theFile.readline()
            while len(_row) > 3:
                _row = _row.strip().translate(str.maketrans({',':None, '[':None, ']':None}))
                junk = _row.split()
                junk = [int(junk[0]), [float(x) for x in junk[1:]]]
                _parmlist = _theDict[junk[0]]
                _theDict[junk[0]] = _parmlist + junk[1:] + [np.max(junk[1:])] + [np.argmax(junk[1:])]
                _row = _theFile.readline()
            if len(_row) <= 3:
                _valAccFlag = False
                _trainLossFlag = True

        if (_trainLossFlag) and (_row.find('Training Loss:') != -1):
            _row = _theFile.readline()
            while len(_row) > 3:
                _row = _row.strip().translate(str.maketrans({',':None, '[':None, ']':None}))
                junk = _row.split()
                junk = [int(junk[0]), [float(x) for x in junk[1:]]]
                _parmlist = _theDict[junk[0]]
                _theDict[junk[0]] = _parmlist + junk[1:] + [np.min(junk[1:])] + [np.argmin(junk[1:])]
                _row = _theFile.readline()
            if len(_row) <= 3:
                _trainLossFlag = False
                _valLossFlag = True

        if (_valLossFlag) and (_row.find('Validation Loss:') != -1):
            _row = _theFile.readline()
            while len(_row) > 3:
                _row = _row.strip().translate(str.maketrans({',':None, '[':None, ']':None}))
                junk = _row.split()
                junk = [int(junk[0]), [float(x) for x in junk[1:]]]
                _parmlist = _theDict[junk[0]]
                _theDict[junk[0]] = _parmlist + junk[1:] + [np.min(junk[1:])] + [np.argmin(junk[1:])]
                _row = _theFile.readline()
            if len(_row) <= 3:
                _valLossFlag = False

        if _row.find('Results:') != -1:
            _pandasFlag = False
            _trainAccFlag = True
        _row = _theFile.readline()

for key, value in _theDict.items():
    print(key)
    post_row(_cursor, 'cnn_results', value)
_db.commit()

_db.close()
