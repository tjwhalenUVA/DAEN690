#! /Users/dpbrinegar/anaconda3/envs/gitHub/bin/python
# -*- coding: utf-8 -*-

import argparse                         # package for handling command line arguments

def main(_dbFile):

    import sqlite3
    import numpy as np
    from pandas import DataFrame

    _dartCount = 1000

    # Create connection to the database file
    _db = sqlite3.connect(_dbFile)
    _cur = _db.cursor()

    # Load the data from the database
    _command = "SELECT cn.id, ln.bias_final, cn.text " + \
               "FROM train_content cn, train_lean ln " + \
               "WHERE (cn.id < 9999999999) AND " + \
               "(cn.`published-at` >= '2009-01-01') AND " + \
               "(cn.id == ln.id) AND " + \
               "(ln.url_keep == 1) AND " + \
               "(cn.id NOT IN (SELECT a.id " + \
               "FROM train_content a, train_content b " + \
               "WHERE (a.id < b.id) AND " + \
               "(a.text == b.text)));"

    _cur.execute(_command)
    _df = DataFrame(_cur.fetchall(), columns=('id', 'lean', 'text'))
    _db.close()
    print('%s records read from database' % len(_df))

    _leanValuesDict = {'left': 0,
                       'left-center': 1,
                       'least': 2,
                       'right-center': 3,
                       'right': 4}
    _leanVals = np.array([[_leanValuesDict[k] for k in _df.lean],] * _dartCount)

    print([np.sum([np.random.randint(low=0, high=5, size=len(_df)) == row for row in _leanVals])][0] / len(_df) / 1000.0)


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='Random dart throwing model.')
    _parser.add_argument('dbfile', type=str, default='articles_zenodo.db',
                         help='Path/file to the article database.')
    _args = _parser.parse_args()

    print(_args)

    main(_args.dbfile)
