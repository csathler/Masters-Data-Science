#!/bin/sh

function usage
{
   echo "Usage: deploy.sh <project_base_dir> <java_home>"

}

if [ "$2" = "" ];
then
    usage;
    exit 1
fi
I590TwitterDataset=$1
JAVA_HOME=$2
cd $I590TwitterDataset
mkdir bin
for i in $(ls templates|grep template); do
script=bin/$i
script=`echo $script|sed 's/.template/.sh/g'`
echo "#!/bin/bash
JAVA_HOME=$JAVA_HOME
I590TwitterDataset=$I590TwitterDataset
" > $script
cat templates/$i >> $script
chmod 744 $script
done
