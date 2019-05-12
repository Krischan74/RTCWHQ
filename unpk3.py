import os
import stat
import sys
import zipfile

path=sys.argv[1]

print("----------------------------------------------------------------------")
print("Unpacking PK3s to separate folders in "+path)
print("----------------------------------------------------------------------")

for path, dir_list, file_list in os.walk(path):

	for file_name in file_list:
	
		if file_name.endswith(".pk3"):
		
			print("UnPK3 "+file_name)
			
			abs_file_path = os.path.join(path, file_name)

			parent_path = os.path.split(abs_file_path)[0]
			output_folder_name = os.path.splitext(file_name)[0]
			output_path = os.path.join(parent_path, output_folder_name)

			# create ZIP file
			zip_obj = zipfile.ZipFile(abs_file_path, 'r')
			zip_obj.extractall(output_path)
			zip_obj.close()

			# make PK3 writable and delete it
			os.chmod(abs_file_path,stat.S_IWRITE)
			os.remove(abs_file_path)