import csv
from datetime import datetime
from csv import DictReader
import gc
import time
import sys

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

D = 2 ** 25
data_path = "data/"


print("Content..")
with open(data_path + "promoted_content.csv") as infile:
    prcont = csv.reader(infile)
    #prcont_header = (prcont.next())[1:]
    prcont_header = next(prcont)[1:]
    prcont_dict = {}
    for ind,row in enumerate(prcont):
        prcont_dict[int(row[0])] = row[1:]
        if ind%100000 == 0:
            print "Events : ", ind
        # if ind==10000:
        #     break
    print(len(prcont_dict))
del prcont

print("Events..")
with open(data_path + "events.csv") as infile:
    events = csv.reader(infile)
    #events.next()
    next(events)
    event_header = [ 'uuid', 'platform', 'loc_country', 'loc_state', 'loc_dma' ]
    event_dict = {}
    for ind,row in enumerate(events):
        tlist = row[ 1:2 ] + row[ 4:5 ]
        loc = row[5].split('>')
        if len(loc) == 3:
            tlist.extend(loc[:])
        elif len(loc) == 2:
            tlist.extend( loc[:]+[''])
        elif len(loc) == 1:
            tlist.extend( loc[:]+['',''])
        else:
            tlist.extend(['','',''])
        event_dict[int(row[0])] = tlist[:] 
        if ind%100000 == 0:
            print "Events : ", ind
        # if ind==10000:
        #     break
    print(len(event_dict))
del events


print("Leakage file..")
leak_uuid_dict= {}

with open(data_path+"leak.csv") as infile:
    doc = csv.reader(infile)
    next( doc )
    leak_uuid_dict = {}
    for ind, row in enumerate(doc):
        doc_id = int(row[0])
        leak_uuid_dict[doc_id] = set(row[1].split(' '))
        if ind%100000==0:
            print "Leakage file : ", ind
    print(len(leak_uuid_dict))
del doc

gc.collect()

alltrain = []

for t, row in enumerate(DictReader(open(data_path+'clicks_train.csv'))):
    # process id
    disp_id = int(row['display_id'])
    ad_id = int(row['ad_id'])

    # process clicks
    y = 0.
    if 'clicked' in row:
        if row['clicked'] == '1':
            y = 1.
        del row['clicked']

    x = []
    for key in row:
        x.append(abs(hash(key + '_' + row[key])) % D)

    row = prcont_dict.get(ad_id, [])
    # build x
    doc_id = -1
    for ind, val in enumerate(row):
        if ind==0:
            doc_id = int(val)
        x.append(abs(hash(prcont_header[ind] + '_' + val)) % D)

    row = event_dict.get(disp_id, [])
    ## build x
    uuid_val = -1
    for ind, val in enumerate(row):
        if ind==0:
            uuid_val = val
        x.append(abs(hash(event_header[ind] + '_' + val)) % D)

    if (doc_id in leak_uuid_dict) and (uuid_val in leak_uuid_dict[doc_id]):
        x.append(abs(hash('leakage_row_found_1'))%D)
    else:
        x.append(abs(hash('leakage_row_not_found'))%D)

    x.append( y )

    alltrain.append( x )
    del x
    if t%100000==0:
        print "Train : ", t
        sys.stdout.flush()

gc.collect()

train = open( data_path + 'train', 'w' )
valid = open( data_path + 'valid', 'w' )
validtrain = open( data_path + 'validtrain', 'w' )
sampletrain = open( data_path + 'sampletrain', 'w' )

for t, x in enumerate( alltrain ):
    row = str( x[-1] ) + ' '
    for field, feature in enumerate( x[:-1] ):
        row = row + str( field + 1 ) + ':' + str( feature ) + ':1 '
    row = row + '\n'
    train.write( row )
    if t < len( alltrain ) / 6 * 5:
        if t % 100 == 0:
            valid.write( row )
        elif t % 100 == 1:
            sampletrain.write( row )
        else:
            validtrain.write( row )
    else:
        if t % 20 == 0:
            valid.write( row )
        elif t % 100 == 1:
            sampletrain.write( row )
        else:
            validtrain.write( row )

gc.collect()

train.close()
valid.close()
validtrain.close()
sampletrain.close()

test = open( data_path + 'test', 'w' )

for t, row in enumerate(DictReader(open(data_path+'clicks_test.csv'))):
    # process id
    disp_id = int(row['display_id'])
    ad_id = int(row['ad_id'])

    # process clicks

    x = []
    for key in row:
        x.append(abs(hash(key + '_' + row[key])) % D)

    row = prcont_dict.get(ad_id, [])
    # build x
    doc_id = -1
    for ind, val in enumerate(row):
        if ind==0:
            doc_id = int(val)
        x.append(abs(hash(prcont_header[ind] + '_' + val)) % D)

    row = event_dict.get(disp_id, [])
    ## build x
    uuid_val = -1
    for ind, val in enumerate(row):
        if ind==0:
            uuid_val = val
        x.append(abs(hash(event_header[ind] + '_' + val)) % D)

    if (doc_id in leak_uuid_dict) and (uuid_val in leak_uuid_dict[doc_id]):
        x.append(abs(hash('leakage_row_found_1'))%D)
    else:
        x.append(abs(hash('leakage_row_not_found'))%D)


    row = '0 '
    for field, feature in enumerate( x[:-1] ):
        row = row + str( field + 1 ) + ':' + str( feature ) + ':1 '
    row = row + '\n'
    test.write( row )
    del x
    del row
    if t%100000==0:
        print "Test : ", t
        sys.stdout.flush()

test.close()