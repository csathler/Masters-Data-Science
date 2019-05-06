#!/bin/bash
function usage
{
   echo "Usage: reformat.sh <input file> <output file>"

}

if [ "$2" = "" ];
then
    usage;
    exit 1
fi


iconv -f ISO-8859-1 -t UTF-8 $1 > $2
