# README file of I590-TwitterDataSet Project
# Author: Yuan Luo <yuanluo@indiana.edu>


1. check the build.properites, make necessary change
2. Build and deply the code
        $ ant
3. Run the code
        Reformat given txt file from ISO-8859-1 to UTF-8 format, run:
        
	$ ./bin/reformat.sh <input file> <output file>

        To import the data, run the following script.
        
	$ ./bin/import_mangodb.sh <db name> <collection name> <import file type> <import file>
	
	Now you can run the query, using a query criteria file (query json file), and update the user profile collection. 
	A working query criteria is included in the input/query.json
	
	$ ./bin/QueryAndUpdate.sh < configuration file> <db name> <collection name> <query json file> <log file>

	If you change the code, you can rebuild and deploy with ant. Here is a script to help you run any new java Class. 
	
	$./bin/ClassRun.sh  <Main Class> <Arguments>

4. Note:
	Before importing a data file, make sure it contains a headerline.

	
5. Visualization:
	Issue a query against the user collection in the database and export the result in csv format. You need to specify  <db name>, <collection name> and <output file name> in the following sample query_and_export command. What the following command does is export formatted addresses and user names of all users whose locations contain both "IN" and "USA".

	$mongoexport -d <db name> -c <collection name> -q "{\"geocode\": {\$exists: true,\$ne:null}, \"geocode.formatted_address\":{\$regex: \"USA\"},\"geocode.formatted_address\":{ \$regex: \"IN\" }}" --csv  --fields geocode.formatted_address,user_name -o <output file name>

	Reformat the output csv file above to meet the visualization format.

	$ awk '{ printf("[ %s ],\n", $l);}' <output file name>

	Now the screen output can be used to create visualization html file. 
	See html sample here: https://developers.google.com/chart/interactive/docs/gallery/map

