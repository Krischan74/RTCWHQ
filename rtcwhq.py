import sys                          # LIB: System
import os.path, stat                # LIB: Path, File status
import glob                         # LIB: Global
import cv2                          # LIB: OpenCV
import numpy as np                  # LIB: Numpy
import torch                        # LIB: Pytorch
import architecture as arch         # LIB: ERSGAN architecture
import subprocess                   # LIB: Call Subprocess
import pathlib                      # LIB: Pathlib
import time                         # LIB: Time
from PIL import Image               # LIB: PIL
from PIL import ImageEnhance        # LIB: PIL Enhancement
from PIL import ImageFilter         # LIB: PIL Filters
from os.path import splitext        # LIB: extension split

# Changeable flags
powertwo = True                     # check for and correct textures which are not power of two size
rtcwexcludes = True                 # exclude defined RTCW/ET folders and use standard settings there
alphaoptimize = True                # use gaussian blur, contrast and brightness (or not, if not needed)
usesharpen = True                   # sharpen the high resolution texture before resize to increase quality
autoconvert = True                  # convert the image to RGB if it is NOT RBG/RGBA!
skiptracemap = True                 # don't resize / include the tracemap
scalelightmaps = True               # resize Lightmaps (could look better, could look strange)
scalelarge = False                  # scale large images too (True = they are initially resized to a lower res)
testmode = False                    # in Testmode, a Lancosz method is used instead of the ESRGAN method
warnings = False                    # ignore (False) or show (True) warnings

# VRAM limits 8GB
largelimit = 2048*2048              # maximum texture scaling limit (stop scaling if texture is below this size)
vramlimit = 1024*512                # maximum size a texture can have before scaling that no CUDA error occurs
									# this depends on available VRAM size, 1024*512 ist for 8GB VRAM, an
									# approx. calculation of the VRAM usage is: width*height*8192 in Bytes
									# so 1024*512*8192 = ~ 4.2GB plus the VRAM already used by Windows/Apps
							
# Predefined Values
modelfactor = 4                     # the scale the selected model has been trained on (default is 4x)
allowed = [".png",".tga",".jpg"]    # allowed image file extensions to process (default: PNG, TGA, JPG)
scaling = Image.LANCZOS             # scaling method reducing too large images for next scale pass
finishing = Image.LANCZOS           # scaling method reducing the highres image to the desired resolution
target = 'cuda'                     # ESRGAN target device: 'cuda' for nVidia card (fast) or 'cpu' for ATI/CPU

# Predefined ESRGAN Models
model_path = 'models/cartoonpainted_400000.pth'       # default model
font_model_path='models/ReducedColorsAttempt.pth'     # font model

# RTCW exclude files (not implemented yet)
excludes = ["gfx/2d/backtile.jpg"]  # ET: gives a strange background texture in the loading screen, should be black

# create logfile
log=open("convert.log","w+")

# ignore warning
if(warnings==False):
	import warnings
	warnings.filterwarnings("ignore")

# function: write to logfile and output to stdout
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

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)
	
# "resize" the image until it fits in vram, return the width (aspect is known)
def fitimage(width, height, vramlimit):

	# reduce image size by factor 2
	while(width*height>vramlimit):
		width=int(width/2)
		height=int(height/2)

	write_log("  - Image resized to "+str(width)+"x"+str(height))
		
	return width
	
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
write_log("Model:           {:s}".format(model_path))
write_log("Fontmodel:       {:s}".format(font_model_path))
write_log("Folder:          " + input)
write_log("Maximum scaling: " + str(factor) + "x")
write_log("Maximum size:    " + str(maxsize) + " Pixel")
write_log("Gaussian Blur:   " + str(blur) + " Pixel")
write_log("Contrast:        " + str(int(contrast*100)) + "%")
write_log("Brightness:      " + str(int(brightness*100)) + "%")
write_log("Sharpen:         " + str(sharpen) + " Pixel")
write_log("JPEG Quality:    " + str(jpegquality))
write_log("----------------------------------------------------------------------")

# count files to process
pngCounter = sum(1 for f in pathlib.Path(input).glob('**/*.png'))
tgaCounter = sum(1 for f in pathlib.Path(input).glob('**/*.tga'))
jpgCounter = sum(1 for f in pathlib.Path(input).glob('**/*.jpg'))
Counter=pngCounter+tgaCounter+jpgCounter
write_log("Found "+str(Counter)+" images: "+str(pngCounter)+" PNG, "+str(tgaCounter)+" TGA and "+str(jpgCounter)+" JPEG")

# init model
if(testmode==False):
	write_log("----------------------------------------------------------------------")
	write_log("Puny human is instructed to wait until the models are been prepared...")

	device = torch.device(target)

	# prepare model
	model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
	model.load_state_dict(torch.load(model_path), strict=True)
	model.eval()

	for k, v in model.named_parameters():
		v.requires_grad = False
		model = model.to(device)
	
	write_log("Model ready.")

	# prepare font model
	fontmodel = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
	fontmodel.load_state_dict(torch.load(font_model_path), strict=True)
	fontmodel.eval()

	for k, v in fontmodel.named_parameters():
		v.requires_grad = False
		fontmodel = fontmodel.to(device)

	write_log("FontModel ready. Let's go!")

starttime=time.time()
cnt=0
dcnt=0
	
# iterate through all subfolders
for dirName, subdirList, fileList in os.walk(input, topdown=False):

	for fname in fileList:

		dirName.replace("/","\\")
		fname.replace("/","\\")
		path=dirName+"/"+fname
		
		delete=False
		delreason=""
		
		# split filename, make it writable and convert extension to lowercase
		os.chmod(path ,stat.S_IWRITE)
		filename,ext=splitext(fname)
		ext=ext.lower()
		
		# only convert allowed file extensions
		if ext in allowed:
			
			# open image, add extension to filename and save it to PNG
			fullname=dirName + "\\" + filename + ext
			
			# try to load the image or throw error if there is a problem
			try:
				im = Image.open(fullname)
				loaded=True
				width, height = im.size
				width=int(width)
				height=int(height)
				imode=im.mode
				cnt+=1

			# image could not open? error!
			except IOError:
				loaded=False
				delete=True
				delreason="could not open image"
				im.close()
			
			# skip tracemap if set
			if(skiptracemap==True and "_tracemap" in filename):
				loaded=False
				delete=True
				delreason="tracemaps not allowed"
				im.close()
						
			# skip lightmaps if set
			if(scalelightmaps==False and "lm_0" in filename):
				loaded=Fals
				delete=True
				delreason="lightmaps not allowed"
				im.close()
						
			# loading successful? check colormode first!
			if(loaded==True):
						
				write_log("----------------------------------------------------------------------")
				write_log("IMAGE "+str(cnt)+" of "+str(Counter)+": "+fullname)
				write_log("----------------------------------------------------------------------")
				stime=time.time()
				
				add="unknown"
				if(imode=="L"):
					add="RGB 8bit Greyscale"
				
				if(imode=="P"):
					add="RGB 8bit Palette"

				if(imode=="RGB"):
					add="RGB 24bit"
					
				if(imode=="RGBA"):
					add="RGB 32bit with Alpha Channel"
				
				write_log("- Colormode: "+add)

				# image is NOT RGB(A)? then try to convert it
				if(autoconvert==True and imode != 'RGBA' and imode != 'RGB'):
				
					# 8bit color palette/greyscale? then convert to 24bit RGB
					if(imode=="P" or imode=="L"):
						im2=Image.new("RGB",im.size)
						im2.paste(im)
						im=im2
						im2=None
						write_log("- NOTICE: 8bit converted to 24bit RGB")
					else:
						# image was not valid, skip
						loaded=False
						delete=True
						delreason="no valid image"
						im.close()
						
			# still a valid image? then process it
			if(loaded==True):
				
				# optional: check and correct textures which are NOT power of two size (can cause errors otherwise)
				if(powertwo):

					twow=poweroftwo(width)
					twoh=poweroftwo(height)
					
					# width/height doesn't match with calculated power of two value? correct it = resize to next power of two
					if (twow != width) or (twoh != height):
						write_log("- NOTICE: Texture size "+str(width)+"x"+str(height)+" corrected - was NOT power of two!")
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
				
				write_log("- Resolution: "+str(width)+"x"+str(height)+" ("+str(factor)+"x = "+str(width*factor)+"x"+str(height*factor)+")")
				
				# initial check if image is already too large: reduce it before enlargement
				process=True
				if(scalelarge==True):
					if((width*height)>vramlimit):
						write_log("  - Image too large, must be resized first")
						
						aspect=width/height
						width=fitimage(width,height,vramlimit)
						height=int(width/aspect)				

						im=im.resize((width,height),scaling)
						if(alpha):
							alpha=alpha.resize((width,height),Image.LANCZOS)
				# simple check if the image needs processing or not
				else:
					if(width*height>=maxsize*maxsize):
						write_log("- no processing: image is already High Resolution")
						process=False
						
				# assume we're rescaling the image
				rescale=True
				
				# process images if necessary
				if(process==True):
					
					# resize image until it is large enough (change the largelimit var to a lower value if it crashes)
					while(width*height<largelimit and rescale==True):
					
						write_log("- Scale Pass #"+str(scalepass))

						# -------------------------------------------------------------------------------
						# Image is too large to scale? reduce size by factor two until the size is valid
						# according to VRAM limit.
						#
						# Notice: ESRGAN uses a lot of GPU VRAM so there is a limitation for the input
						# depending on the available VRAM. An estimated maximum value for 8GB VRAM is
						# about 1024x512 = 524288 Pixels, so change the vramlimit variable if you have
						# more or less than 8GB VRAM. An approximate formula to calculate the size:
						#
						# VRAM needed = width*height*8192 in Bytes
						#
						# You must add to this value the VRAM already assigned by Windows and Apps, which
						# can already take 2-3GB on 8GB VRAM, consider this
						# -------------------------------------------------------------------------------
						if((width*height)>vramlimit):
											
							write_log("  - Image doesn't fit in VRAM, must be resized")
							
							aspect=width/height
							width=fitimage(width,height,vramlimit)
							height=int(width/aspect)
							
							im=im.resize((width,height),scaling)
							if(alpha):
								alpha=alpha.resize((width,height),Image.LANCZOS)

						# scale color image
						width=width*modelfactor
						height=height*modelfactor
						write_log("  - ERSGAN scales Colormap to "+str(width)+"x"+str(height))
						if(testmode==False):
							if(("font" or "hudchars") in dirName):
								im=Image.fromarray(upscale(im,device,fontmodel))
							else:
								im=Image.fromarray(upscale(im,device,model))
						else:
							im=im.resize((width*modelfactor,height*modelfactor),scaling)
						
						# scale alpha channel
						if(alpha):
							write_log("  - ERSGAN scales Alphamap to "+str(width)+"x"+str(height))
							if(testmode==False):
								if(("font" or "hudchars") in dirName):
									alpha=Image.fromarray(upscale(alpha,device,fontmodel))
								else:
									alpha=Image.fromarray(upscale(alpha,device,model))
							else:
								alpha=alpha.resize((width*modelfactor,height*modelfactor),scaling)
							
						# image has target size? don't rescale anymore
						if(width==(ow*factor) and height==(oh*factor)):
							rescale=False
						# otherwise do the next scale pass
						else:
							scalepass+=1
												
					# calculate final texture size
					nsw=int(ow*factor)
					nsh=int(oh*factor)
					ms=maxsize
					bl=blur
					co=contrast
					br=brightness
					sh=sharpen
					
					# rtcw/et excludes and default settings for specific folder names
					if(rtcwexcludes):

						# limit general font size to 1024 and use different values for blur/contrast/brightness/sharpen
						if("font" in dirName):
							ms=1024
							sh=4
							bl=1
							co=2.0
							br=-0.5
							write_log("- Font Texture found, limiting size to "+str(ms)+" Pixel")
							
						# limit ET HUD font size to 1024 and use different values for blur/contrast/brightness/sharpen
						if("hudchars" in filename):
							ms=1024
							sh=4
							bl=0
							co=4.0
							br=-0.5
							write_log("- Font Texture found, limiting size to "+str(ms)+" Pixel")
							
						# limit leveshots image size to 1024
						if("levelshots" in dirName):
							ms=512
							# but the survey map can still be large
							if("_cc") in filename:
								ms=maxsize
							write_log("- Levelshot Texture found, limiting size to "+str(ms)+" Pixel")
						
						# limit lightmaps image size to 1024
						if("maps" in dirName):
							ms=1024
							write_log("- Lightmap Texture found, limiting size to "+str(ms)+" Pixel")
						
						# dont' add contrast to the skies and user different blur value
						folders=["skies","sfx","liquids"]
						if(dirName in folders):
						#if(("skies" in dirName) or ("sfx" in dirName) or ("liquids" in dirName)):
							co=0.0
							br=0.0
							bl=2
							write_log("- Blurry Alpha Texture found - no contrast or brightness change!")


					# optional: sharpen filter
					if(usesharpen and sh!=0):
						
						# don't sharpen lightmaps!
						if ("maps" in dirName):				
							write_log("- Lightmap texture found, no sharpen on lightmaps")
						else:
							write_log("- Colormap: Sharpen")
							im=ImageEnhance.Sharpness(im).enhance(sh)
							
					# if texture size is too large? reduce by factor
					if(nsw>ms) or (nsh>ms):
						f=min(ms/nsw,ms/nsh)
						sw=int(nsw*f)
						sh=int(nsh*f)
						write_log("- Colormap: scaled to "+str(sw)+"x"+str(sh)+" ("+str(nsw)+"x"+str(nsh)+" is too large)")
					# or use the calculated values
					else:
						sw=nsw
						sh=nsh
						write_log("- Colormap: scaled to "+str(sw)+"x"+str(sh))
								
					# scale colormap to desired resolution
					im=im.resize((sw,sh),finishing)
						
					# scae alphamap, if there is alpha
					if(alpha):
						
						# convert alphamap to 8bit greyscale to increase processing speed (it's greyscale only)
						alpha = alpha.convert("L")
										
						# only perform this if alpha optimizations are desired
						if(alphaoptimize):
						
							# apply gaussian blur
							if(bl!=0):
								write_log("- Alphamap: Gaussian Blur")
								alpha = alpha.filter(ImageFilter.GaussianBlur(bl))
								
							# apply brightness
							if(br!=0.0):
								write_log("- Alphamap: Brightness")
								alpha = ImageEnhance.Brightness(alpha).enhance(1.0+br)

							# apply contrast
							if(co!=0.0):
								write_log("- Alphamap: Contrast")
								alpha = ImageEnhance.Contrast(alpha).enhance(1.0+co)
						
						# scale alpha to desired resolution
						write_log("- Alphamap scaled to "+str(sw)+"x"+str(sh))
						alpha=alpha.resize((sw,sh),finishing)

						# merge alpha channel with RGB
						write_log("- Merging Colormap with Alphamap")
						im.putalpha(alpha.split()[-1])
					
					# save file
					write_log("- Replacing original Texture")
					im.save(fullname,quality=jpegquality,optimize=True,progressive=True)
					im.close()
					write_log("- Conversion completed in "+str(hms_string(time.time()-stime)))
				
		else:
			# remove all other files
			delete=True

		# delete files if flagged for deletion
		if(delete==True):
			write_log("----------------------------------------------------------------------")
			write_log("- DELETED "+path+" ("+delreason+")")
			os.remove(path)
			dcnt+=1
			
# finish
write_log("----------------------------------------------------------------------")
write_log("Removing empty directories...")
remove_empty_dirs(input)
write_log("Converted " + str(cnt) + " images and deleted "+str(dcnt)+" other files in "+str(hms_string(time.time()-starttime))+". Done.")
log.close()

# wait for input and exit
subprocess.call('timeout /T 5')