<img src="https://i.imgur.com/mRyCGM0.jpg">

No more blurry textures. Never again. This tool / script collection is for upscaling textures and graphics in the old games **Return to Castle Wolfenstein (RTCW)** and **Wolfenstein: Enemy Territory (ET)** using [ESRGAN]( https://github.com/xinntao/ESRGAN) - and possibly other games, too. It contains three useful Python scripts for automated scaling a folder of images while preserving the Alphachannel of the 32bit TGA files. It runs from a **single command file**.

I've upscaled my whole RTCW/ET textures and it looks great now, it took about 40 minutes to convert all of the RTCW and ET textures here on a GTX2080 8GB (non-Ti). The results are **stunning** and I love to play and walk around looking at two complete new nearly 20 year old games to me :) Now you can rescale them on your own needs.

But you can use RTCWHQ to **upscale any kind of Images, Icons, Textures, Pixelart** or whatever. There are many pretrained models out there and you can even train your own models to achieve even better results - but this is rocket science to me. I've been happy with the model provided in the repository as the results are already sufficient to me.

If you're in hurry you can download the whole project files and my zipped Python installation here:
<a href="http://www.mediafire.com/file/il3h07mmpana2bb/Python37.zip/file">Python37.zip</a> (910MB) <a href="http://www.mediafire.com/file/368mwaimrej2ae7/RTCWHQ.zip/file">RTCWHQ.zip</a> (241MB)

# Table of contents

- [Features](#features)
- [Quick Usage Instructions](#quickusage)
- [Screenshots](#screenshots)
	- [Wolfenstein: Enemy Territory comparison](#et)
	- [Return to Castle Wolfenstein comparison](#rtcw)
	- [Texture comparison](#textures)
	- [Examples RTCW](#examples)
- [Python Setup](#python)
- [Batch Workflow](#workflow)

- [Known Issues](#issues)  
	- [General Issues](#generalissues)  
	- [RTCW Issues](#rtcwissues)  

- [Tweaking](#tweaking)  
 	- [Flags](#flags)
 	- [Variables](#variables)
	- [Best practice](#bestpractice)
	- [Gassian Blur (Alpha channel)](#blur)
	- [Contrast and Brightness (Alpha channel)](#contrast)
 	- [Sharpen (Color channels)](#contrast)
	- [Autoexcludes (RTCW/ET specific)](#autoexcludes)

- [Changelog](#changelog)  


<a name="features"></a>

# Features
- **Python-only** solution
- **converts** your RTCW/ET/Q3 whatever textures from Lores to Hires using ESRGAN
- **customizable**: change scale, limits, blur, contrast, brightness, output quality
- fully **automated**, just copy one or many PK3(s) in the input folder and run the batch
- upscales the **alpha channel** in high quality, too!
- outputs a detailed **Logfile**
- designed for **Windows** but may run on Linux, too


<a name="quickusage"></a>

# Quick Usage Instructions
- install Python like [described here](#python)
- get [ETlegacy](https://www.etlegacy.com) and install ET - it's free (and [RTCW](https://store.steampowered.com/) is still available on Steam)
- put one or more PK3s in the "input" folder (custom maps, pak0.pk3 and so on)
- open a CMD window in the root of the RTCWHQ folder (where the convert.cmd is located)
- adjust the settings in the batch file if needed (see [Tweaking](#tweaking) )
- run the "convert.cmd" batch file
- copy the PK3 back to the game
- take a look at the [Known Issues](#issues) to fix problems
- have fun!


<a name="screenshots"></a>

# Screenshots
Some Screenshots and Textures from both games for direct comparison. See the difference with the Alpha layers. All original textures are Copyright (C) 1999-2010 id Software LLC, a ZeniMax Media company and only used/altered here to show the difference. RTCW and ET are still two of my most favourite games and the creators did a very good job, thanks!

<a name="et"></a>

### Wolfenstein: Enemy Territory Comparison
Screenshots from the Battery Map. Left: original (LQ) and Right: 4x scaled (HQ). The difference may be small but sometimes it really looks stunning, not so blurry and more sharpened. Try it yourself, it is hard to see the effect in a single Screenshot.

<img src="https://i.imgur.com/N3n75lc.jpg" width="384"> <img src="https://i.imgur.com/wxQzrni.jpg" width="384">
<img src="https://i.imgur.com/3PDkpjn.jpg" width="384"> <img src="https://i.imgur.com/hJsxMcz.jpg" width="384">
<img src="https://i.imgur.com/EW7H09t.jpg" width="384"> <img src="https://i.imgur.com/f8fhKL9.jpg" width="384">
<img src="https://i.imgur.com/rCgm208.jpg" width="384"> <img src="https://i.imgur.com/MaLReZk.jpg" width="384">
<img src="https://i.imgur.com/s7EdnM9.jpg" width="384"> <img src="https://i.imgur.com/pR9eHrB.jpg" width="384">
<img src="https://i.imgur.com/4AT7aDY.jpg" width="384"> <img src="https://i.imgur.com/CzrSDxr.jpg" width="384">
<img src="https://i.imgur.com/gAaLR15.jpg" width="384"> <img src="https://i.imgur.com/Po6tK55.jpg" width="384">
<img src="https://i.imgur.com/EtrYlXX.jpg" width="384"> <img src="https://i.imgur.com/MzWfRsS.jpg" width="384">
<img src="https://i.imgur.com/wpE1OiY.jpg" width="384"> <img src="https://i.imgur.com/LTDyNxk.jpg" width="384">
<img src="https://i.imgur.com/YitiADm.jpg" width="384"> <img src="https://i.imgur.com/NZXlNFo.jpg" width="384">
<img src="https://i.imgur.com/DH1x5RL.jpg" width="384"> <img src="https://i.imgur.com/Flk8jZD.jpg" width="384">
<img src="https://i.imgur.com/VsE11uU.jpg" width="384"> <img src="https://i.imgur.com/nmP2y8Q.jpg" width="384">
<img src="https://i.imgur.com/vDqHRKa.jpg" width="384"> <img src="https://i.imgur.com/35mI3wd.jpg" width="384">


<a name="rtcw"></a>

### Return to Castle Wolfenstein Comparison
Screenshots from the Chateau SP Map. Left: original (LQ) and Right: 4x scaled (HQ). The difference may be small but sometimes it really looks stunning, not so blurry and more sharpened. Try it yourself, it is hard to see the effect in a single Screenshot.

<img src="https://i.imgur.com/3Un8r7l.jpg" width="384"> <img src="https://i.imgur.com/J2B9Urf.jpg" width="384">
<img src="https://i.imgur.com/LIMAbAv.jpg" width="384"> <img src="https://i.imgur.com/nxYzsbQ.jpg" width="384">
<img src="https://i.imgur.com/liCgt0D.jpg" width="384"> <img src="https://i.imgur.com/u1247bb.jpg" width="384">
<img src="https://i.imgur.com/e6mkl2r.jpg" width="384"> <img src="https://i.imgur.com/xT7y2xp.jpg" width="384">
<img src="https://i.imgur.com/PLTEhYB.jpg" width="384"> <img src="https://i.imgur.com/jxpgqkT.jpg" width="384">
<img src="https://i.imgur.com/z7Yhc4m.jpg" width="384"> <img src="https://i.imgur.com/6pJapXH.jpg" width="384">
<img src="https://i.imgur.com/pV8dxNj.jpg" width="384"> <img src="https://i.imgur.com/HAK9cOV.jpg" width="384">
<img src="https://i.imgur.com/CR7fmsA.jpg" width="384"> <img src="https://i.imgur.com/FmdT8eN.jpg" width="384">
<img src="https://i.imgur.com/6hhaDDx.jpg" width="384"> <img src="https://i.imgur.com/eGCqphK.jpg" width="384">
<img src="https://i.imgur.com/m9w4f9P.jpg" width="384"> <img src="https://i.imgur.com/fp48tb3.jpg" width="384">
<img src="https://i.imgur.com/pl6JxGH.jpg" width="384"> <img src="https://i.imgur.com/DFuUsgQ.jpg" width="384">
<img src="https://i.imgur.com/jk5Q11y.jpg" width="384"> <img src="https://i.imgur.com/Ky6fcm4.jpg" width="384">
<img src="https://i.imgur.com/i68J7zn.jpg" width="384"> <img src="https://i.imgur.com/bpsFZof.jpg" width="384">
<img src="https://i.imgur.com/AxR7zFt.jpg" width="384"> <img src="https://i.imgur.com/djTyulJ.jpg" width="384">
<img src="https://i.imgur.com/xEqWOuA.jpg" width="384"> <img src="https://i.imgur.com/QF0Ab8l.jpg" width="384">
<img src="https://i.imgur.com/7THJFoc.jpg" width="384"> <img src="https://i.imgur.com/t9juRnI.jpg" width="384">

<a name="textures"></a>

### Texture Comparison
Some single textures to compare. Left: the 4x scaled Texture, Right: a GIF animation before/after scaling. I think you'll see the difference and why ESRGAN is so superior compared to other scaling algorithms. Click the left image to see the texture in the scaled resolution.

<img src="https://i.imgur.com/bAoXzAR.png" width="384"> <img src="https://i.imgur.com/D07NMll.gif" width="384">
<img src="https://i.imgur.com/SoO8Xej.png" width="384"> <img src="https://i.imgur.com/SRLcB2U.gif" width="384">
<img src="https://i.imgur.com/av4IG3H.png" width="384"> <img src="https://i.imgur.com/bGau5En.gif" width="384">
<img src="https://i.imgur.com/Ke97kFV.png" width="384"> <img src="https://i.imgur.com/4LgLVmT.gif" width="384">
<img src="https://i.imgur.com/mhpDjTd.png" width="384"> <img src="https://i.imgur.com/qqeBagZ.gif" width="384">
<img src="https://i.imgur.com/CU3lg9C.jpg" width="384"> <img src="https://i.imgur.com/Mvkbe1k.gif" width="384">
<img src="https://i.imgur.com/AlIi1mM.png" width="384"> <img src="https://i.imgur.com/FppUy8T.gif" width="384">
<img src="https://i.imgur.com/C9SCYcj.png" width="384"> <img src="https://i.imgur.com/UJ6j8ei.gif" width="384">


<a name="examples"></a>

### Examples RTCW
Some Ultra Widescreen ingame Screenshots (3840x1600) from different SP maps with the converted textures using this tool, default settings. I've only added some of my own custom blood textures.

<img src="https://i.imgur.com/wdx2bvL.jpg" width="768">

<img src="https://i.imgur.com/ZHHqBGT.jpg" width="384"> <img src="https://i.imgur.com/FPLXGch.jpg" width="384">
<img src="https://i.imgur.com/v4cXgJh.jpg" width="384"> <img src="https://i.imgur.com/H1iJPnM.jpg" width="384">
<img src="https://i.imgur.com/xNppXw2.jpg" width="384"> <img src="https://i.imgur.com/C9i2pVM.jpg" width="384">
<img src="https://i.imgur.com/V39j4C8.jpg" width="384"> <img src="https://i.imgur.com/XhjzdyY.jpg" width="384">
<img src="https://i.imgur.com/rKNCwyy.jpg" width="384"> <img src="https://i.imgur.com/RfnXuRe.jpg" width="384">
<img src="https://i.imgur.com/PpxJNBV.jpg" width="384"> <img src="https://i.imgur.com/M9ANL9e.jpg" width="384">
<img src="https://i.imgur.com/8gjJHIu.jpg" width="384"> <img src="https://i.imgur.com/gNucRBZ.jpg" width="384">
<img src="https://i.imgur.com/XOTmJqc.jpg" width="384"> <img src="https://i.imgur.com/iktOyzJ.jpg" width="384">
<img src="https://i.imgur.com/coM5WHE.jpg" width="384"> <img src="https://i.imgur.com/fOZDXEH.jpg" width="384">
<img src="https://i.imgur.com/qTlrqxc.jpg" width="384"> <img src="https://i.imgur.com/aKDtk0k.jpg" width="384">
<img src="https://i.imgur.com/95aYxak.jpg" width="384"> <img src="https://i.imgur.com/ZKCLAr8.jpg" width="384">
<img src="https://i.imgur.com/jgiI1gw.jpg" width="384"> <img src="https://i.imgur.com/xDtUQ3v.jpg" width="384">


<a name="python"></a>

# Python Setup
First you must install Python and add some libs to it before you can run the scripts. I've tested it and it works this way:

|order|action|
|:-:|:-|
|1|download and install [Python 3.7](https://www.python.org/downloads/release/python-373/) (use the Windows x86-64 executable installer which adds the PATH variables)|
|2|install the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive)|
|3|go to the [Pytorch website](https://pytorch.org/get-started/locally/) and select for example: Stable/Windows/PIP/Python 3.7/CUDA 10.0|
|4|run the two commands shown below the selection box in a commandline window|
|5|example: **pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl**|
|6|example: **pip3 install torchvision**|
|7|run this command too: **pip3 install numpy opencv-python**|

This installs Python, pytorch, numpy and opencv and all dependencies which are necessary to run RTCWHQ. All other libs should be part of the Python package, make sure you do a full install (you won't need the Debug libs), and click the Add to Path box to run Python from everywhere.

Though it is a very complex process with many single steps I tried to automate RTCWHQ as much as possible. The scripts are also designed in such a way that you can call individual procedures directly if necessary and even tweak the settings.

I've used the following packages/commands for developing/testing RTCWHQ:
1. python-3.7.3-amd64.exe
2. pip3 install numpy-1.16.3-cp37-cp37m-win_amd64.whl
3. pip3 install torch-1.1.0-cp37-cp37m-win_amd64.whl
4. pip3 install Pillow-6.0.0-cp37-cp37m-win_amd64.whl
5. pip3 install six-1.12.0-py2.py3-none-any.whl
6. pip3 install torchvision-0.2.2.post3-py2.py3-none-any.whl
7. pip3 install opencv_python-4.1.0.25-cp37-cp37m-win_amd64.whl


<a name="workflow"></a>

# Batch Workflow
This is a long description of the single steps the batch command calls and executes. Like I already said, it is very complex but good to know what is happening here.

1. **Execute unpk3.py**
	- This Python script unpacks all PK3s from the input folder into separate folders with the same name like the PK3 below the input folder
   
2. **Execute rtcwhq.py**
	- loads the ESRGAN model "cartoonpainted_400000.pth" taken from here: [https://esrgan.blogspot.com/2019/01/blog-post.html](https://esrgan.blogspot.com/2019/01/blog-post.html)
	- loads the ESRGAN model "ReducedColorsAttempt.pth" tkane from here: [https://kingdomakrillic.tumblr.com/post/181294654011/a-collection-of-great-art-oriented-models-not](https://kingdomakrillic.tumblr.com/post/181294654011/a-collection-of-great-art-oriented-models-not)
	- parses the input folder for PNG, JPG or TGA files
	- counts the numbers of images there
	- removes readonly attributes from these files
	- too large images are resized to their half resolution before upscaling (if flag is set)
	- images not following power of two resolution are size-corrected, if flag is set
	- trying to convert images with different color depth (not RGB/RGBA) to RGB
	- special RTCW/ET folders not worth to rescale are ignored if a flag is set
	- the scaling runs in a loop until the maximum resolution or the target resolution has been reached
	- the Alpha channel is scaled, too!
	- after scaling, a slight Gaussian Blur is applied to the Alpha channel to sharpen the edges
	- additional, Contrast and Brightness changes are applied to the Alpha channel to have more control over the edge contrast
	- a sharpen filter is applied to the Color channels
	- the image is then saved in its original format (PNG, TGA, JPG)
	- TGA with Alpha is saved as uncompressed TGA32
	- TGA without Alpha is saved as uncompressed TGA24
	- JPG is saved als JPEG (Quality 90 default)
	- deletes **ALL** other files, just keeping the scaled images and the folder structure
	- removes **ALL** empty folders which are not needed anymore
	- writes a log to "convert.log" to let you check for errors if something goes wrong

3. **Execute makepk3.py**
	- parses the input folder for folders in its root = the scaled, separate PK3 archive contents as a folder
	- ZIP each of these folders back to a PK3 file and add "_HQ" to the filename
	- optional you can define there if you want to delete the folders and just keep the new PK3s (default is keep folders)

You can then take these PK3 files and copy them at the folder where the original PK3 is located. RTCW and ET will recognize that the first part of the filename belongs to the original PK3, checks which files are newer and uses the newer files instead. So basically they are a mixture of all files but our scaled images will be used instead of the old Lowres images resulting in a new game experience. Sometimes you must add "Z_" or "ZZZ_" to the filename as the alphabetical order is important which file is checked last, but this depends on the filename of the PK3 or if other PK3s are used, like a script patch.


<a name="issues"></a>

# Known issues

<a name="generalissues"></a>

### General Issues
- the "cartoonpainted_400000.pth" model is a factor 4x only upscaling model, consider this!
- the "ReducedColorsAttempt.pth" model is a factor 4x only upscaling model, consider this!
- you're an ATI rebel and don't like nVidia? Bad luck - ESRGAN is based on CUDA, though it can run very slow on a CPU - set the variable **target** from **cuda** to **cpu** to switch
- ESRGAN uses a lot of GPU VRAM so there is a limitation for input scale, an estimated maximum value for 8GB VRAM is 1024x512 = 524288 Pixels
- input images above this size are resized by factor 2 until they are below this limit, if the flag is set!
- this is because it will result in errors if you try to enlarge a large texture, it must stay below this limit unless you have dozens of GB of VRAM
- Alpha textures can sometimes make trouble - most of them are scaled properly with the default settings but sometimes you must tweak them later or delete from from the source PK3s, see Tweaking for more details
- Loading times. Even with ETlegacy the loading times increase a lot with huge textures. You should really run ET from a M.2 SSD with four PCIe lanes :-)
- you'll need a lot of VRAM to play RTCW/ET with HQ textures, scaling to 2048 Pixels consume approximately 2-3GB of VRAM on a single map, RTCW/ET may even crash
- sometimes I've experienced an endless loop with the ESRGAN resize/scale loop but I couldn't find the problem, to avoid this make sure you're converting ONLY textures with a power of two resolution (2^x = 128x128, 128x256, 512x512, 1024x512 and so on)
- some custom maps MAY contain invalid images, I've found textures with very unusual resolutions or even PCX images named as TGA
- rtcwhq tries to convert them but I haven't test all maps available out there so expect the unexpected (though the standard PK3s should convert without any issues)

<a name="rtcwissues"></a>

### RTCW won't start with HQ textures
- Update your RTCW to the [unoffical 1.42 patch](https://wolffiles.de/index.php?filebase&fid=4905) to get sp_pak4.pk3
- use the great [IORTCW 1.51C](http://wolfenstein4ever.de/index.php/downloads/viewdownload/7/2495) to bypass the loading errors
- adjust the wolfconfig.cfg in the c:\Users\[User]\Documents\RTCW\main\ folder:


<a name="tweaking"></a>

# Tweaking
You can adjust the Alpha channels that they get more sharp edges. While most of the Alpha textures are scaled properly some need extra treatment. For example, a diffuse puddle texture looks better with more blurry edges while a fence needs a hard edge. So how do we get good scaled Alpha channels? Blur them!

You can set the blur, contrast and brightness values in my batch file. Take a look at this line with the default settings:

> Syntax: python.exe rtcwhq.py [path] [scale] [size] [blur] [contrast] [brightness] [sharpen] [quality]

> Default: python.exe rtcwhq.py input 8 2048 2 2.0 0.0 4 90

**Argument details:**

|arg|default value|valid range|description|
|:-:|:-:|:-:|:-|
|1|input|valid path (string)|check the (currentdir)/input/ folder for images to convert|
|2|8|2-8 (integer)|scale by a maximum factor of 8|
|3|2048|512-2048 (integer)|with a maximum of 2048 pixel (longest side) in the target texture|
|4|2|0-X (integer)|apply 2 Pixel Gaussian Blur to the Alphamap|
|5|2.0|0.0 to 5.0 (float)|apply 200% contrast to the Alphamap|
|6|0.0|-1.0 to 1.0 (float)|apply 0% brightness to the Alphamap|
|7|4|0 to 8 (integer)|apply a Sharpen Filter to the Colormap with Intensity 4|
|8|90|1-100 (integer)|JPEG save quality (1=worst, 100=best, 90=barely visible artifacts)|

<a name="flags"></a>

### Flags
In the **rtcwhq.py** there are some flags you can set to **True** (enabled) or **False** (disabled):

|flag|default|description|
|:-|:-:|:-|
|powertwo|True|check for and correct textures which are not power of two size|
|rtcwexcludes|True|exclude defined RTCW/ET folders and use standard settings there (see below)|
|alphaoptimize|True|use gaussian blur, contrast and brightness (or not, if not needed)|
|usesharpen|True|sharpen the high resolution texture before resize to increase quality|
|autoconvert|True|convert the image to RGB if it is NOT RBG/RGBA!|
|skiptracemap|True|don't resize / include the tracemap|
|scalelightmaps|True|resize Lightmaps (could look better, could look strange)|
|scalelarge|False|scale large images too (True = they are initially resized to a lower res)|
|testmode|False|in Testmode, a Lancosz method is used instead of the ESRGAN method|
|warnings|False|ignore (False) or show (True) warnings|

<a name="variables"></a>

### Variables
In the **rtcwhq.py** there are some default variables you can change if needed but only if you know what you're doing here:

|var|default|description|
|:-|:-:|:-|
|largelimit|2048*2048|maximum texture size limit for scaling|
|vramlimit|1024*512|maximum size a texture can have before scaling that no CUDA error occurs, this depends on available VRAM size, 1024*512 ist for 8GB VRAM, 5GB used|
|modelfactor|4|the scale the selected model has been trained on (default is 4x)|
|allowed|[".png",".tga",".jpg"]|allowed image file extensions to process (default: PNG, TGA, JPG)|
|scaling|Image.LANCZOS|scaling method reducing too large images for next scale pass|
|finishing|Image.LANCZOS|scaling method reducing the highres image to the desired resolution|
|target|'cuda'|ESRGAN target device: 'cuda' for nVidia card (fast) or 'cpu' for ATI/CPU|
|model_path|'models/cartoonpainted_400000.pth'|the path to the model you want to use for textures|
|font_model_path|'models/ReducedColorsAttempt.pth'|the path to the model you want to use for font texture scaling|

Valid scaling methods (PIL) are: **Image.NEAREST**, **Image.BILINEAR**, **Image.BICUBIC** or **Image.LANCZOS** only

<a name="bestpractice"></a>

### Best practice
The best practice is to copy all the PK3 you want to enlarge at once in the input folder and run the default settings. After the script has finished, there are PK3s with the new textures for each of the subfolders. Move them away from the input folder. Now check the images one by one - delete the images which are ok and keep all others. Overwrite them with the original image and start another run with different settings. Copy the new files over the ones in the new PK3s.

<a name="blur"></a>

### Gaussian Blur (Alpha Channel)
The Gaussian Blur filter softens the pixelated alpha edges. I found this trick playing with Photoshop and it works quite good in combination with a contrast change but I've been tired to do all the clicks in Photoshop so I decided to write a neat tool to do this for me. As a rule of thumb, the larger the original texture, the smaller the blur amount. Because RTCWHQ only blurs the largest possible image after resize there is not much to change here only if your maximum output resolution is for example 512 only.

<a name="contrast"></a>

### Contrast and Brightness (Alpha Channel)
To get the sharp edges back again we must play with the brightness and contrast of the Alpha channel. A good start is to increase the contrast by 200% (2.0), the more contrast the sharper the edge. Sometimes we must decrease the brightness too and increase the contrast - so we can "move" the edge a little bit. This works vice versa by increasing the brightness while maintaining the higher contrast. You should avoid to use a negative contrast as it makes no sense.

<a name="sharpen"></a>

### Sharpen (Color Channels)
The input image gets resized to a higher resolution until the set maximum limit has been reached and if set, a sharpen filter is applied to it before it gets resized to its final output resolution, which in most cases should be one power of two level below. The sharpening increases the image quality but may cause artifacts, so be careful with too high values here.

<a name="autoexcludes"></a>

### Autoexcludes (RTCW/ET specific)
This tool is written for RTCW/ET but can be used for other purposes, too. I've added some RTCW/ET specific checks, please consider this and switch the Flag **rtcwexcludes** in **rtcwhq.py** to **False** if it causes trouble:
- files inside folders containing "fonts" in the folder name get a maximum texture size of 1024 (fonts)
- the hudchars.tga file gets an extra setting to create clean edges
- files inside folders containing "levelshots" in the folder name get a maximum texture size of 512 (except the _cc map)
- files inside folders containing "maps" in the folder name get a maximum texture size of 1024 (lightmaps)
- files inside folders containing "skies", "sfx" or "liquids" in the folder name get NO contrast or brightness change (blurry alphas)


<a name="changelog"></a>

# Changelog

- 05/14/2019 Minor update: small fixes
  - fixed a small bug in the poweroftwo function
  - replaced all Screenshots with better examples

- 05/12/2019 Major update: many extensions to rtcwhq.py
  - added flags: autoconvert, skiptracemap, scalelightmaps, scalelarge, testmode, warnings
  - added texture size fix "poweroftwo" and conversion of non RGB(A) files to RGB (there were some problems with flughafen.pk3 and PCX files named as TGA which resulted in fatal errors, now the script converts or skips them
  - added a testmode to save time (it only scales with Lanczos instead of ESRGAN, to test settings and for debugging)
  - did a complete test run with RTCW/ET, some RTCW MP/SP custom maps and about 60 ET custom maps:
  - Benchmark: ET pak0.pk3 with 1449 images, converted to 4x or max. 2048 Pixels in ~ 13 Minutes (1.4GB final PK3 size)
  - Benchmark: RTCW pak0.pk3 with 2164 images, converted to 4x or max. 2048 Pixels in ~ 27 Minutes (0.7GB final PK3 size)
  - Benchmark performed on a i7-9700K@3.6GHz (8 real cores) / nVidia RTX2080 8GB (Inno3D)

- 05/06/2019 Updated Readme
  - added some information how to bypass problems in RTCW with HQ textures
  - added animated comparison GIFs
  - Readme updated

- 05/05/2019 Switched over to Python
  - the new python script includes all features and gives better results
  - got rid of the Blitzmax tool

- 05/04/2019 Initial commit
  - the first version relies on a additional tool written in Blitzmax
