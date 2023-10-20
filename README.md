# Intelligent terrain generation considering global information and terrain patterns

Simulated terrains can provide rich information for landform and terrain research, disaster prediction, 
rescue and disaster relief, and national security. Quickly generating an accurate simulated terrain for 
target areas is of great importance. However, for existing data-driven terrain generation methods, it is 
difficult to balance modeling accuracy and the amount of data required. To overcome this gap, this study 
proposes a deep learning method that integrates global information and pattern features of the local 
terrain (IGPN) to realize terrain generation. In our proposed IGPN method, we first apply both terrain 
patterns (ridge line and drainage line) and elevation points as input data; thus, the terrain features 
(i.e., channel and peak) and elevation of the target terrain can be provided. The local information 
extraction module (LIEM) and global information extraction module (GIEM) are then applied to generate 
local and global terrain features, respectively. Thus, global and local terrain features can be supplemented,
and a more accurate terrain can be generated. Experiments show that the IGPN method performs state-of-the-art
terrain-generation tasks. Specifically, compared with existing terrain generation methods (IETA, SRResNet, 
Bicubic, Kriging, FEN, and ERFFN), the terrain generated by IGPN is closer to the real terrain and can retain
more local terrain features. 
Keywords: Simulated terrain; local information; Kriging; terrain features; elevation point


Code availability section
Name of the code: Terrain Generation.
Contact: mxc@cug.edu.cn
Program language: Python.
Software required: Python 3.8, TensorFlow 2.0, pykrige.ok library.
The source codes are available for downloading at the link:
https://github.com/MXCCUG/Terrain-generation
Data availability
The data are available (https://figshare.com/s/a61678f0df2df498abf0).




