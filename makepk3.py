import os
import sys
import zipfile
import shutil

# variables
delete=False      # delete folder after zipping (True or False)

# add files/folders to a zip archive
def zipfolder(path, relname, archive):
	paths = os.listdir(path)
	for p in paths:
		p1 = os.path.join(path, p) 
		p2 = os.path.join(relname, p)
		if os.path.isdir(p1): 
			print("Zipping " + p1)
			zipfolder(p1, p2, archive)
			if(delete):
				shutil.rmtree(p1, ignore_errors=True)
		else:
			archive.write(p1, p2) 

# create a zip archive
def create_zip(path, relname, archname):
	archive = zipfile.ZipFile(archname, "w", zipfile.ZIP_DEFLATED)
	if os.path.isdir(path):
		zipfolder(path, relname, archive)
	else:
		archive.write(path, relname)
	archive.close()
	
path = sys.argv[1]
paths = os.listdir(path)

print("----------------------------------------------------------------------")
print("Creating PK3s from the folders in the root of "+path)
print("----------------------------------------------------------------------")

for p in paths:
	
	# take only the root folder inside the path
	pk = os.path.join("", p)

	# only ZIP folders
	if(os.path.isdir(path+"/"+pk)):
		print("Moving files to " + pk + ".pk3")
		create_zip(path + "/" + pk,"",path + "/" + pk + "_HQ.pk3")
		if(delete):
			shutil.rmtree(path + "/" + pk, ignore_errors = True)