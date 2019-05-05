#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Author: Carlos Sathler
# cssathler@gmail.com
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#####################################################################################
# Download data for the project
# Researchers who made the data available request we cite this reference:
#@inproceedings{tan+etal:16a, 
#   author = {Chenhao Tan and Vlad Niculae and Cristian Danescu-Niculescu-Mizil and Lillian Lee}, 
#   title = {Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions}, 
#   year = {2016}, 
#   booktitle = {Proceedings of WWW} 
#}
import sys, os
import wget
import tarfile
import bz2

#------------------------------------------------------------------------------------
# Checks if data file needs to be downloaded
#
def needs_download(fname):
    if not os.path.exists(fname):
        return 1 
    else:
        return 0

#------------------------------------------------------------------------------------
# Extracts json (list) file 
#
def decompress(source, target):

    print "Decompressing " + source
    f = open(source, 'r')
    x = f.read()
    y = bz2.decompress(x)
    f.close()
    
    print "Saving " + target 
    f = open(target,'w')
    f.write(y)
    f.close()

#------------------------------------------------------------------------------------
# Return input file names 
#
def get_input_file_names():
    return ("all/train_period_data.jsonlist", "all/heldout_period_data.jsonlist")

#------------------------------------------------------------------------------------
# Extracts json (list) file 
#
def download_files(url, fname):

    #----------------------------------
    # Begin citation 1.1 
    #
    # These few lines of code draw variable names and tarfile functionality from 
    # https://vene.ro/blog/winning-arguments-attitude-change-reddit-cmv.html
    # Author: Vlad Niculae
    #

    full_url = url + fname
    train_fname_bz2 = "all/train_period_data.jsonlist.bz2"
    heldout_fname_bz2 = "all/heldout_period_data.jsonlist.bz2"
    
    print "Downloading file " + full_url
    wget.download( full_url, bar=None )
    
    f = open(fname, 'rb')
    tar = tarfile.open(fileobj=f, mode="r")

    #
    # End citation 1.1
    #----------------------------------
   
    print "Extracting training data: " + train_fname_bz2
    tar.extract(train_fname_bz2)

    print "Extracting test data: " + heldout_fname_bz2
    tar.extract(heldout_fname_bz2)
    f.close()

    print "Decompressing files..."
    (train_fname, heldout_fname) = get_input_file_names()
    decompress(train_fname_bz2, train_fname)
    decompress(heldout_fname_bz2, heldout_fname)
  
    return (train_fname, heldout_fname)
    
#------------------------------------------------------------------------------------
# testing... 
#
if __name__ == '__main__':

    fname = "cmv.tar.bz2"
    url = "https://chenhaot.com/data/cmv/"
    if needs_download(fname): 
        (train_data, test_data) = download_files(url, fname)
