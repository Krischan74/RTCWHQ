import sys                          # LIB: System
import os.path, stat                # LIB: Path, File status
import glob                         # LIB: Global
import cv2                          # LIB: OpenCV
import numpy as np                  # LIB: Numpy
import torch                        # LIB: Pytorch
import architecture as arch         # LIB: ERSGAN architecture
import subprocess                   # LIB: Call Subprocess
import pathlib                      # LIB: Pathlib
from PIL import Image               # LIB: PIL
from PIL import ImageEnhance        # LIB: PIL Enhancement
from PIL import ImageFilter         # LIB: PIL Filters
from os.path import splitext        # LIB: extension split

import warnings
warnings.filterwarnings("ignore")

# function: create logfile
log=open("convert.log","w+")
def write_log(*args):
	line = ' '.join([str(a) for a in args])
	log.write(line+'\n')
	print(line)

# function: delete a single directory
def remove_empty_dir(path):
	try:
		if(os.rmdir(path)):
			write_log("Removed: " + path)
	except OSError:
		write_log("Not removed: " + path)
		pass

# function: delete a directory tree
def remove_empty_dirs(path):
	for root, dirnames, filenames in os.walk(path, topdown=False):
		for dirname in dirnames:
			remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))

# function: upscale a PNG image
def upscale(im, device, model):
	img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
	img = img * 1.0 / 255
	img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
	img_LR = img.unsqueeze(0)
	img_LR = img_LR.to(device)
	output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
	output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
	output = (output * 255.0).round()
	output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)	
	return output.astype(np.uint8)

# get the next power of two value of a value (texture size correction)
def poweroftwo(val):
	val=int(val)
	# check range from 2^0=1 to 2^15 = 32768 (should be large enough :-)
	for i in range(0,15):
		# value is below current power of two? found!
		if val<pow(2,i):
			# get previous power of two and next power of two
			mn=pow(2,i-1)
			mx=pow(2,i)
			# get delta between previous and next power (middle)
			delta=(mx-mn)/2
			# value above the middle: use higher power of two else use lower power of two
			if val>(mn+delta):
				return mx
			else:
				return mn
	
# Changeable flags
powertwo = True                  # check for and correct textures which are not power of two size
rtcwexcludes = True              # exclude defined RTCW/ET folders and use standard settings there
alphaoptimize = True             # use gaussian blur, contrast and brightness (or not, if not needed)
usesharpen = True                # sharpen the high resolution texture before resize to increase quality

# VRAM limits 8GB
largelimit = 2048*2048           # maximum texture size limit
vramlimit = 1024*512             # maximum size a texture can have before scaling that no CUDA error occurs
                                 # this depends on available VRAM size, 1024*512 ist for 8GB VRAM, 5GB used
							
# Predefined Values
modelfactor = 4                  # the scale the selected model has been trained on (default is 4x)
allowed = [".png",".tga",".jpg"] # allowed image file extensions to process (default: PNG, TGA, JPG)
scaling = Image.LANCZOS          # scaling method reducing too large images for next scale pass
finishing = Image.LANCZOS        # scaling method reducing the highres image to the desired resolution
target = 'cuda'                  # ESRGAN target device: 'cuda' for nVidia card (fast) or 'cpu' for ATI/CPU
model_path = 'models/cartoonpainted_400000.pth'

# commandline input
input = sys.argv[1]
factor = sys.argv[2]
maxsize = sys.argv[3]
blur = sys.argv[4]
contrast = sys.argv[5]
brightness = sys.argv[6]
sharpen = sys.argv[7]
jpegquality = sys.argv[8]

# set data types
input=str(input)
factor=int(factor)
maxsize=int(maxsize)
blur=int(blur)
contrast=float(contrast)
brightness=float(brightness)
sharpen=int(sharpen)
jpegquality=int(jpegquality)

# start
write_log("======================================================================")
write_log("RTCWHQ batch upscaling with ERSGAN started")
write_log("======================================================================")
write_log("Model:         {:s}".format(model_path))
write_log("Folder:        " + input)
write_log("Scale factor:  " + str(factor) + "x")
write_log("Maximum size:  " + str(maxsize) + " Pixel")
write_log("Gaussian Blur: " + str(blur) + " Pixel")
write_log("Contrast:      " + str(int(contrast*100)) + "%")
write_log("Brightness:    " + str(int(brightness*100)) + "%")
write_log("Sharpen:       " + str(sharpen) + " Pixel")
write_log("JPEG Quality:  " + str(jpegquality))
write_log("----------------------------------------------------------------------")
write_log("Puny human is instructed to wait until the model has been prepared...")

# count files to process
pngCounter = sum(1 for f in pathlib.Path(input).glob('**/*.png'))
tgaCounter = sum(1 for f in pathlib.Path(input).glob('**/*.tga'))
jpgCounter = sum(1 for f in pathlib.Path(input).glob('**/*.jpg'))
Counter=pngCounter+tgaCounter+jpgCounter
write_log("Found "+str(Counter)+" images: "+str(pngCounter)+" PNG, "+str(tgaCounter)+" TGA and "+str(jpgCounter)+" JPEG")

# init model
device = torch.device(target)
model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()

# prepare model
for k, v in model.named_parameters():
	v.requires_grad = False
	model = model.to(device)

cnt=0
dcnt=0
write_log("Model ready. Let's go!")

# iterate through all subfolders
for dirName, subdirList, fileList in os.walk(input, topdown=False):

	for fname in fileList:

		dirName.replace("/","\\")
		fname.replace("/","\\")
		path=dirName+"/"+fname
		
		# split filename, make it writable and convert extension to lowercase
		os.chmod(path ,stat.S_IWRITE)
		filename,ext=splitext(fname)
		ext=ext.lower()
		
		# only convert allowed file extensions
		if ext in allowed:
			
			# file counter
			cnt+=1

			# open image, add extension to filename and save it to PNG
			fullname=dirName + "\\" + filename + ext
			im = Image.open(fullname)
			width, height = im.size
			width=int(width)
			height=int(height)
			
			write_log("----------------------------------------------------------------------")
			write_log("PROCESSING FILE "+str(cnt)+" of "+str(Counter)+": "+fullname)
			write_log("----------------------------------------------------------------------")
						
			# optional: check and correct textures which are NOT power of two size (can cause errors otherwise)
			if(powertwo):

				twow=poweroftwo(width)
				twoh=poweroftwo(height)
				
				# width/height doesn't match with calculated power of two value? correct it = resize to next power of two
				if (twow != width) or (twoh != height):
					write_log("- ERROR corrected: Texture size "+str(width)+"x"+str(height)+" was NOT power of two!")
					width=twow
					height=twoh
					im.resize((width,height),Image.LANCZOS)
			
			# store original width for later use
			ow=width
			oh=height

			# if exists, get alpha channel first before we mess with the original image
			alpha=None
			if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
				alpha=im.split()[-1]
				# convert to RGB to work with ESRGAN
				alpha = alpha.convert("RGB")
				
			scalepass=1
			
			# resize image until it is large enough (change the 2048 values to a lower value if crash)
			while((width*height)<largelimit):
			
				write_log("Scale Pass #"+str(scalepass))
				write_log("- original size is "+str(width)+"x"+str(height))

				# image is too large to scale? reduce size with nearest neighbour first!
				# -------------------------------------------------------------------------------
				# Notice: ESRGAN uses a lot of GPU VRAM so there is a limitation for input scale.
				# An estimated maximum value for 8GB VRAM is 1024x512 = 524288 Pixels, so change
				# this if you have more or less VRAM than in my example here
				# -------------------------------------------------------------------------------
				if((width*height)>=vramlimit):
									
					write_log("- image too large, must be resized")
					
					# reduce image size by factor 2
					width=int(width/2)
					height=int(height/2)
					write_log("- resize to "+str(width)+"x"+str(height))
					im=im.resize((width,height),scaling)
					if(alpha):
						alpha=alpha.resize((width,height),Image.LANCZOS)

				# scale color image
				width=width*modelfactor
				height=height*modelfactor
				write_log("- ERSGAN scale color to "+str(width)+"x"+str(height))
				im=Image.fromarray(upscale(im,device,model))
				
				# scale alpha channel
				if(alpha):
					write_log("- ERSGAN scale alpha to "+str(width)+"x"+str(height))
					alpha=Image.fromarray(upscale(alpha,device,model))
					
				scalepass+=1
				
			# optional: sharpen filter
			if(usesharpen):
				
				# apply sharpen filter
				if(sharpen!=0):
					
					# don't sharpen lightmaps!
					if ("maps" in dirName):				
						write_log("- Lightmap texture found, no sharpen on lightmaps")
					else:
						write_log("- apply sharpen filter")
						im=ImageEnhance.Sharpness(im).enhance(sharpen)
					
			# calculate final texture size
			nsw=int(ow*factor)
			nsh=int(oh*factor)
			ms=maxsize
			co=contrast
			br=brightness
			
			# rtcw/et excludes and default settings for specific folder names
			if(rtcwexcludes):

				# limit font size to 1024
				if("fonts" in dirName):
					ms=1024
					write_log("- Font texture found, limiting size to "+str(ms)+" Pixel")

				# limit leveshots image size to 1024
				if("levelshots" in dirName):
					ms=512
					# but the survey map can still be large
					if("_cc") in filename:
						ms=maxsize
					write_log("- Levelshot texture found, limiting size to "+str(ms)+" Pixel")
				
				# limit lightmaps image size to 1024
				if("maps" in dirName):
					ms=1024
					write_log("- Lightmap texture found, limiting size to "+str(ms)+" Pixel")
				
				# dont' add contrast to the skies
				if(("skies" in dirName) or ("sfx" in dirName) or ("liquids" in dirName)):
					co=0.0
					br=0.0
					write_log("- blurry Alpha texture found - no contrast or brightness change!")

			# if texture size is too large? reduce by factor
			if(nsw>ms) or (nsh>ms):
				fw=ms/nsw
				fh=ms/nsh
				f=min(fw,fh)
				sw=int(nsw*f)
				sh=int(nsh*f)
				write_log("- output texture too large ("+str(nsw)+"x"+str(nsh)+", reducing to "+str(sw)+"x"+str(sh)+")")
			else:
				sw=nsw
				sh=nsh
						
			# scale color to desired resolution
			write_log("- scale color to "+str(sw)+"x"+str(sh))
			im=im.resize((sw,sh),finishing)
				
			# treat alpha, if alpha
			if(alpha):
				
				# convert alpha to 8bit greyscale to increase processing speed
				alpha = alpha.convert("L")

				# scale alpha to desired resolution
				write_log("- scale alpha to "+str(sw)+"x"+str(sh))
				alpha=alpha.resize((sw,sh),finishing)
								
				# only perform this if alpha optimizations are desired
				if(alphaoptimize):
				
					# apply gaussian blur
					if(blur!=0):
						write_log("- apply gaussian blur filter")
						alpha = alpha.filter(ImageFilter.GaussianBlur(blur))
						
					# apply brightness
					if(brightness!=0.0):
						write_log("- apply brightness filter")
						alpha = ImageEnhance.Brightness(alpha).enhance(1.0+br)

					# apply contrast
					if(contrast!=0.0):
						write_log("- apply contrast filter")
						alpha = ImageEnhance.Contrast(alpha).enhance(1.0+co)
				
				# merge alpha channel with RGB
				write_log("- merge color with alpha")
				im.putalpha(alpha.split()[-1])
			
			# save file
			write_log("- replace original texture")
			im.save(fullname,quality=jpegquality,optimize=True,progressive=True)
			im.close()
			
		else:
			# remove all other files
			os.remove(path)
			write_log("Removed: "+path)
			dcnt+=1
# finish
write_log("----------------------------------------------------------------------")
write_log("Removing empty directories...")
remove_empty_dirs(input)
write_log("Converted " + str(cnt) + " images and deleted "+str(dcnt)+" other files. Done.")
log.close()

# wait for input and exit
subprocess.call('timeout /T 5')