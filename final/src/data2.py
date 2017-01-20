import csv
import gc


group = []

result = open( 'result', 'w' )
result.write( 'display_id,ad_id\n' )

prev = None

with open('data/'+'clicks_test.csv') as infile:
    test = csv.reader( infile )
    for prob, row in zip( open( 'output', 'r' ), test ):
        if prev != None:
            if row[ 0 ] != prev:
                group.sort( key = lambda x : x[ 1 ] )
                result.write( prev + ',' )
                for a in range( len( group ) - 1 ):
                    result.write( group[ a ][ 0 ] + ' ' )
                result.write( group[ -1 ][ 0 ] + '\n' )
                group = []
                prev = row[ 0 ]
        group.append( [ row[ 1 ], double( prob ) ] )


group.sort( key = lambda x : x[ 1 ] )
result.write( prev + ',' )
for a in range( len( group ) - 1 ):
    result.write( group[ a ][ 0 ] + ' ' )
result.write( group[ -1 ][ 0 ] )

result.close()