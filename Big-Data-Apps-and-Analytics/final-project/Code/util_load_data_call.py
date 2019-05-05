#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Author: Carlos Sathler
# cssathler@gmail.com
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#####################################################################################
# Ensures correctness of call to data_load.py 
#
import sys
import commands as cmd

#------------------------------------------------------------------------------------
# Displays usage and exits 
#
def print_usage_and_exit():
    #(ret, pwd) = cmd.getstatusoutput('pwd')
    print "\nPlease try again..."
    print "Usage: load_data.py all|train|test pos_int_number_of_discussions_trees_to_load"
    print "\nExamples 1: Load all data"
    print "\tpython load_data.py all 50000"
    print "\nExamples 2: Load all train data"
    print "\tpython load_data.py train 50000"
    print "\nExamples 3: Load all test data"
    print "\tpython load_data.py test 50000"
    print "\nExamples 4: Used for testing - Load 100 discussion trees; train data only"
    print "\tpython load_data.py train 100"
    print "\nExamples 5: Used for testing - Load 100 discussion trees; train and test data"
    print "\tpython load_data.py all 100\n"
    exit()

#------------------------------------------------------------------------------------
# Performs a number of tests on parameters used on call and exits if incorrect call
#
def get_good_parms_or_exit(call_line):

    # check correct number of parameters were used in the call
    if len(call_line) <> 3: 
        print_usage_and_exit()
    else:
        (prg, data_to_load, dt_count) = call_line 

    # ensure par 1 is okay
    if data_to_load.upper() not in ('ALL', 'TRAIN', 'TEST'):
        print_usage_and_exit()

    # ensure par 2 is okay
    try: 
        dt_count = int(dt_count)
    except:
        print_usage_and_exit()

    if dt_count <= 0: 
        print_usage_and_exit()

    return (data_to_load.upper(), dt_count)

#------------------------------------------------------------------------------------
# testing....
#
if __name__ == '__main__':

    (data_to_load, dt_count) = get_good_parms_or_exit(sys.argv)
    if data_to_load == 'ALL':
        print "load train data: " + str(dt_count) + " records"
        print "load test data: " + str(dt_count) + " records"
    elif data_to_load == 'TRAIN':
        print "load train data: " + str(dt_count) + " records"
    elif data_to_load == 'TEST':
        print "load test data: " + str(dt_count) + " records"
      
