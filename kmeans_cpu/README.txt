guide for setting up and using the rgb data extraction program
contact griffin for more details if stuff breaks
if you're on linux (mac MIGHT work):
first open up a terminal and go into the jpeg-9a directory in this directory
run the command ./configure
then run the command make
at this point everything should be installed correctly
run command make test 
this should run succesfully and you will see some image files being created

for the program extract_pixels.c
compile it with 
gcc extract_pixels.c -ljpeg -o extract_pixels
on a unix system

ISSUE: extract pixels currently not working on griffin's machine
encountering some kind of library error.