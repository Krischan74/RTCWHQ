import os
import stat
import sys
import zipfile

path=sys.argv[1]

for path, dir_list, file_list in os.walk(path):
	for file_name in file_list:
		if file_name.endswith(".pk3"):
			print("UnPK3 "+file_name)
			abs_file_path = os.path.join(path, file_name)

			# The following three lines of code are only useful if 
			# a. the zip file is to unzipped in it's parent folder and 
			# b. inside the folder of the same name as the file

			parent_path = os.path.split(abs_file_path)[0]
			output_folder_name = os.path.splitext(file_name)[0]
			output_path = os.path.join(parent_path, output_folder_name)

			zip_obj = zipfile.ZipFile(abs_file_path, 'r')
			zip_obj.extractall(output_path)
			zip_obj.close()
			os.chmod(abs_file_path,stat.S_IWRITE)
			os.remove(abs_file_path)