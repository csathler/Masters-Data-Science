#!/bin/bash
function usage
{
   echo "Usage: import_mangodb.sh <db name> <collection name> <import file type> <import file>"

}

if [ "$4" = "" ];
then
    usage;
    exit 1
fi
mongoimport --db $1 --collection $2 --type $3 --headerline --file $4
