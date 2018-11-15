#!/usr/bin/python
#
# Canny edge detection
#
# Use Python 2.7 with these packages: numpy, PyOpenGL, Pillow
#
# Can test this with
#
#   python main.py
#
# This loads 'images/small.png', then applies the 'c' command.
#
# Press '?' to see a list of available commands.  Use + and - to see
# the intermediate stages of the computation.


import sys, os, math

import numpy as np

from PIL import Image, ImageOps

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *


# Globals

windowWidth  = 1000    # window dimensions (not image dimensions)
windowHeight =  800

texID = None           # for OpenGL

zoom = 1.0             # amount by which to zoom images
translate = (0.0,0.0)  # amount by which to translate images


# Image

imageDir      = 'images'
imageFilename = 'small.png'
imagePath     = os.path.join( imageDir, imageFilename )

image          = None    # the image as a 2D np.array
smoothImage    = None    # the smoothed image
gradientMags   = None    # the image with gradient magnitudes (in 0...255)
gradientDirs   = None    # array of gradient directions in [0,7] with direction i = i*45 degrees.
maximaImage    = None    # gradient magnitues with non-maxima set to 0
thresholdImage = None    # thresholded pixels (= 255 or 128 or 0)
edgeImage      = None    # final edges pixels (= 255 or 0)

imageNames = [ 'original image', 'smoothed image', 'gradients', 'gradient directions', 'maxima', 'thresholded maxima', 'Canny edges' ]

currentImage = 0 # the image being displayed

normalizeImage = True # scale image so that its pixels are in the range [0,255]

upperThreshold = 25
lowerThreshold = 5


# Apply Canny edge detection
#
# Returns list of edge pixels

def compute():

  global image, smoothImage, gradientMags, gradientDirs, maximaImage, thresholdImage, edgeImage, currentImage

  height = image.shape[0]
  width  = image.shape[1]

  print 'smoothing'

  if smoothImage is None:
    smoothImage = np.zeros( (height,width), dtype=np.float_ )

  smoothImage = smooth( image )
  print 'finding gradients'

  if gradientMags is None:
    gradientMags = np.zeros((height, width), dtype=np.float_)

  if gradientDirs is None:
    gradientDirs = np.zeros( (height,width), dtype=np.float_ )

  gradientMags, gradientDirs = findGradients( smoothImage )

  print 'suppressing non-maxima'

  if maximaImage is None:
    maximaImage = np.zeros( (height,width), dtype=np.float_ )

  maximaImage = suppressNonMaxima( gradientMags, gradientDirs )

  print 'double thresholding'

  if thresholdImage is None:
    thresholdImage = np.zeros( (height,width), dtype=np.float_ )

  thresholdImage = doubleThreshold( maximaImage )

  print 'edge tracking'

  if edgeImage is None:
    edgeImage = np.zeros( (height,width), dtype=np.float_ )

  edgeImage = trackEdges( thresholdImage )

  # extract edge pixels

  edgePixels = list( np.transpose( np.nonzero( edgeImage ) ) )

  # for debugging: show the image that we're interested in

  currentImage = len(imageNames)-1

  return edgePixels


# Smooth image
#
# Apply the 5x5 filter (below) to 'image' and store the result in
# 'smoothedImage'.
#
# [1 mark]

def smooth(image):

  height = image.shape[0]
  width  = image.shape[1]
  
  kernel = (1/273.0) * np.array( [[1,  4,  7,  4, 1],
                                  [4, 16, 26, 16, 4],
                                  [7, 26, 41, 26, 7],
                                  [4, 16, 26, 16, 4],
                                  [1,  4,  7,  4, 1]] )

  # YOUR CODE HERE
  kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
  smoothedImage = np.zeros_like(image)            # convolution output
  # Add zero padding to the input image
  image_padded = np.zeros((image.shape[0] + 4, image.shape[1] + 4))
  image_padded[2:-2, 2:-2] = image
  for x in range(width):     # Loop over every pixel of the image
    for y in range(height):
      # element-wise multiplication of the kernel and the image
      smoothedImage[y,x]=(kernel*image_padded[y:y+5,x:x+5]).sum()
  return smoothedImage


# Compute the image's gradient magnitudes and directions
#
# The directions are in the range [0,7], where 0 is to the right, 2 is
# up, 4 is left, and 6 is down.  You should *calculate* the direction
# instead of having a big if-then-else.
#
# [2 marks]
#
# 1 of the two marks is for a *good* (i.e. simple, one-line)
# calculation of direction.

def findGradients( image ):

  height = image.shape[0]
  width  = image.shape[1]

  kernel_gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

  kernel_gy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

  # YOUR CODE HERE
  kernel_gx = np.flipud(np.fliplr(kernel_gx))    # Flip the kernel
  Image_gx = np.zeros_like(image)            # convolution output
  Image_gy = np.zeros_like(image)            # convolution output
  # Add zero padding to the input image
  image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
  image_padded[1:-1, 1:-1] = image
  for x in range(width):     # Loop over every pixel of the image
    for y in range(height):
      # element-wise multiplication of the kernel and the image
      Image_gx[y,x] = (kernel_gx * image_padded[y:y + 3, x:x + 3]).sum()
      Image_gy[y,x] = (kernel_gy * image_padded[y:y + 3, x:x + 3]).sum()

  # Calculate the gradient magnitude - Euclidean distance
  gMags = np.sqrt(np.add((Image_gx * Image_gx), (Image_gy * Image_gy)))

  # Find the gradient direction
  # gDirs = np.arctan(Image_gy + 0.001 / (Image_gx + 0.001))  # add 0.001 to avoid dividing by zero
  # gDirs = np.zeros_like(gMags)
  gDirs = np.floor((np.arctan2((Image_gy + 0.001), (Image_gx + 0.001)) + math.pi) / math.pi * 4 ) # add 0.001 to avoid dividing by zero

  return gMags, gDirs
  

# Suppress the non-maxima in the gradient directions
#
# Use the 'offset' array to get the gradient direction for each
# gradient in [0,7].  Do not use a big if-then-else.
#
# [1 mark]

def suppressNonMaxima( gMags, gDirs ):

  # gradient offsets for each gradient direction in [0,7]

  offset = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
  height = gMags.shape[0]
  width = gMags.shape[1]
  maximaIm = np.zeros_like(gMags)

  # YOUR CODE HERE
  for x in range(width):  # Loop over every pixel of the image
    for y in range(height):
      max = gMags[y, x]
      dir = int(gDirs[y, x])  # look up offset index with the direction index

      # compare each pixel against its neighbouring pixels,
      try:
        if max >= gMags[(y + offset[dir][1]), (x + offset[dir][0])] and max >= gMags[(y - offset[dir][1]), (x - offset[dir][0])]:
          maximaIm[y, x] = gMags[y, x]  # if it is a local maxima, set the pixel value
      except IndexError:  # catch IndexError when evaluating boundary pixels and do nothing
        pass

  return maximaIm



# Apply double thresholding
#
# Set pixels < 'lowerThreshold' to 0, pixels above 'upperThreshold' to
# 255, and all other pixels to 128.
#
# [1 mark]

def doubleThreshold( maximaIm ):

  height = maximaIm.shape[0]
  width  = maximaIm.shape[1]

  # YOUR CODE HERE
  thresholdIm = maximaIm

  for x in range(width):     # Loop over every pixel of the image
    for y in range(height):
      if maximaIm[y, x] < lowerThreshold:
          thresholdIm[y, x] = 0
      elif maximaIm[y, x] > upperThreshold:
          thresholdIm[y, x] = 255
      else:
          thresholdIm[y, x] = 128


  return thresholdIm



# Attach weak pixels to strong pixels
#
# Weak pixels = 128.  Strong pixels = 255.  The 'edgePixels' should,
# when done, contain only 0s and 255s.
#
# Use the 'offsets' to find pixels in the neighbourhood of a strong
# pixel.  Use a list of strong pixels, which gets updated as new
# strong pixels are made.
#
# [2 marks]
#
# 1 of the two marks is for an *efficient* implementation.

def trackEdges( thresholdImage ):

  height = thresholdImage.shape[0]
  width  = thresholdImage.shape[1]

  edgeIm = np.zeros_like(thresholdImage)

  offsets = [ (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1) ]

  # YOUR CODE HERE
  for x in range(width):     # Loop over every pixel of the image
    for y in range(height):
      if thresholdImage[y, x] == 255:
        edgeIm[y, x] = 255

        dirc = int(gradientDirs[y, x]) # look up offset index with the direction index
        if thresholdImage[(y+offsets[dirc][1]), (x+offsets[dirc][0])] == 128:
          edgeIm[(y + offsets[dirc][1]), (x + offsets[dirc][0])] = 255
        if thresholdImage[(y-offsets[dirc][1]), (x-offsets[dirc][0])] == 128:
          edgeIm[(y - offsets[dirc][1]), (x - offsets[dirc][0])] = 255

  return edgeIm
    
# File dialog

if sys.platform != 'darwin':
  import Tkinter, tkFileDialog
  root = Tkinter.Tk()
  root.withdraw()



# Set up the display and draw the current image


def display():

  # Clear window

  glClearColor ( 1, 1, 1, 0 )
  glClear( GL_COLOR_BUFFER_BIT )

  glMatrixMode( GL_PROJECTION )
  glLoadIdentity()

  glMatrixMode( GL_MODELVIEW )
  glLoadIdentity()
  glOrtho( 0, windowWidth, 0, windowHeight, 0, 1 )

  # Set up texturing

  global texID
  
  if texID == None:
    texID = glGenTextures(1)

  glPixelStorei( GL_UNPACK_ALIGNMENT, 1 )
  glBindTexture( GL_TEXTURE_2D, texID )

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1,0,0,1] )

  # Images to draw, in rows and columns

  toDraw, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing = getImagesInfo()

  for r in range(rows):
    for c in range(cols):

      # Find lower-left corner
      
      baseX = (horizSpacing + maxWidth ) * c + horizSpacing
      baseY = (vertSpacing  + maxHeight) * (rows-1-r) + vertSpacing

      if toDraw[r][c] is not None:

        img = toDraw[r][c]

        height = scale * img.shape[0]
        width  = scale * img.shape[1]

        # Get pixels and draw

        show = np.real(img)

        # Normalize image so all pixels are in [0,255].  This is useful when debugging because small details are more visible.
        
        if normalizeImage:
          min = np.min(show)
          max = np.max(show)
          if min == max:
            max = min+1
          show = (show - min) / (max-min) * 255

        # Put the image into a texture, then draw it

        imgData = np.array( np.ravel(show), np.uint8 ).tostring()

        # with open( 'out.pgm', 'wb' ) as f:
        #   f.write( 'P5\n%d %d\n255\n' % (img.shape[1], img.shape[0]) )
        #   f.write( imgData )

        glTexImage2D( GL_TEXTURE_2D, 0, GL_INTENSITY, img.shape[1], img.shape[0], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, imgData )

        # Include zoom and translate

        cx     = 0.5 - translate[0]/width
        cy     = 0.5 - translate[1]/height
        offset = 0.5 / zoom

        glEnable( GL_TEXTURE_2D )

        glBegin( GL_QUADS )
        glTexCoord2f( cx-offset, cy-offset )
        glVertex2f( baseX, baseY )
        glTexCoord2f( cx+offset, cy-offset )
        glVertex2f( baseX+width, baseY )
        glTexCoord2f( cx+offset, cy+offset )
        glVertex2f( baseX+width, baseY+height )
        glTexCoord2f( cx-offset, cy+offset )
        glVertex2f( baseX, baseY+height )
        glEnd()

        glDisable( GL_TEXTURE_2D )

        if zoom != 1 or translate != (0,0):
          glColor3f( 0.8, 0.8, 0.8 )
          glBegin( GL_LINE_LOOP )
          glVertex2f( baseX, baseY )
          glVertex2f( baseX+width, baseY )
          glVertex2f( baseX+width, baseY+height )
          glVertex2f( baseX, baseY+height )
          glEnd()

      # Draw image captions

      glColor3f( 0.2, 0.5, 0.7 )
      drawText( baseX, baseY-20, imageNames[currentImage] )

  # Done

  glutSwapBuffers()

  

# Get information about how to place the images.
#
# toDraw                       2D array of images 
# rows, cols                   rows and columns in array
# maxHeight, maxWidth          max height and width of images
# scale                        amount by which to scale images
# horizSpacing, vertSpacing    spacing between images


def getImagesInfo():

  allImages = [ image, smoothImage, gradientMags, gradientDirs, maximaImage, thresholdImage, edgeImage ]

  # Only display a single image

  rows = 1
  cols = 1

  # Find max image dimensions

  maxHeight = 0
  maxWidth  = 0
  
  for img in allImages:
    if img is not None:
      if img.shape[0] > maxHeight:
        maxHeight = img.shape[0]
      if img.shape[1] > maxWidth:
        maxWidth = img.shape[1]

  # Scale everything to fit in the window

  minSpacing = 30 # minimum spacing between images

  scaleX = (windowWidth  - (cols+1)*minSpacing) / float(maxWidth  * cols)
  scaleY = (windowHeight - (rows+1)*minSpacing) / float(maxHeight * rows)

  if scaleX < scaleY:
    scale = scaleX
  else:
    scale = scaleY

  maxWidth  = scale * maxWidth
  maxHeight = scale * maxHeight

  # Draw each image

  horizSpacing = (windowWidth-cols*maxWidth)/(cols+1)
  vertSpacing  = (windowHeight-rows*maxHeight)/(rows+1)

  # only return a single image: the current image

  return [ [ allImages[currentImage] ] ], 1, 1, maxHeight, maxWidth, scale, horizSpacing, vertSpacing
  

  
# Handle keyboard input

def keyboard( key, x, y ):

  global image, imageFilename, smoothImage, gradientMags, gradientDirs, maximaImage, thresholdImage, edgeImage, zoom, translate, currentImage, normalizeImage

  if key == '\033': # ESC = exit
    sys.exit(0)

  elif key == 'i':

    if sys.platform != 'darwin':
      imagePath = tkFileDialog.askopenfilename( initialdir = imageDir )
      if imagePath:

        image = loadImage( imagePath )
        imageFilename = os.path.basename( imagePath )
        currentImage  = 0
        
        smoothImage    = None
        gradientMags   = None
        gradientDirs   = None
        maximaImage    = None
        thresholdImage = None
        edgeImage      = None

  elif key == 'z':
    zoom = 1
    translate = (0,0)

  elif key == 'n':
    normalizeImage = not normalizeImage
    if normalizeImage:
      print 'normalized image'
    else:
      print 'unnormalized image'

  elif key == 'c': # compute
    edgePixels = compute()
    # print 'Edge pixels:'
    # for px in edgePixels:
    #   print ' %.1f,%.1f' % (px[0],px[1])

  elif key in ['+','=']:
    currentImage = (currentImage + 1) % len(imageNames)

  elif key in ['-','_']:
    currentImage = (currentImage - 1 + len(imageNames)) % len(imageNames)

  else:
    print '''keys:
           c  compute the solution
           i  load image
           z  reset the translation and zoom
           +  next image
           -  previous image
    
              translate with left mouse dragging
              zoom with right mouse draggin up/down'''

  glutPostRedisplay()


# Handle special key (e.g. arrows) input

def special( key, x, y ):

  # Nothing done

  glutPostRedisplay()



# Load an image

def loadImage( path ):

  try:
    img = Image.open( path ).convert( 'L' ).transpose( Image.FLIP_TOP_BOTTOM )
    # img = ImageOps.invert(img)
  except:
    print 'Failed to load image %s' % path
    sys.exit(1)

  return np.array( list( img.getdata() ), np.float_ ).reshape( (img.size[1],img.size[0]) )



# Handle window reshape

def reshape( newWidth, newHeight ):

  global windowWidth, windowHeight

  windowWidth  = newWidth
  windowHeight = newHeight

  glViewport( 0, 0, windowWidth, windowHeight )

  glutPostRedisplay()



# Output an image

def outputImage( image, filename ):

  show = np.real(image)

  img = Image.fromarray( np.uint8(show) ).transpose( Image.FLIP_TOP_BOTTOM )

  img.save( filename )




# Draw text in window

def drawText( x, y, text ):

  glRasterPos( x, y )
  for ch in text:
    glutBitmapCharacter( GLUT_BITMAP_8_BY_13, ord(ch) )

    

# Handle mouse click


currentButton = None
initX = 0
initY = 0
initZoom = 0
initTranslate = (0,0)

def mouse( button, state, x, y ):

  global currentButton, initX, initY, initZoom, initTranslate, translate, zoom

  if state == GLUT_DOWN:

    currentButton = button
    initX = x
    initY = y
    initZoom = zoom
    initTranslate = translate

  elif state == GLUT_UP:

    currentButton = None

    if button == GLUT_LEFT_BUTTON and initX == x and initY == y: # Process a left click (with no dragging)

      # Find which image the click is in

      toDraw, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing = getImagesInfo()

      row = (y-vertSpacing ) / float(maxHeight+vertSpacing)
      col = (x-horizSpacing) / float(maxWidth+horizSpacing)

      if row < 0 or row-math.floor(row) > maxHeight/(maxHeight+vertSpacing):
        return

      if col < 0 or col-math.floor(col) > maxWidth/(maxWidth+horizSpacing):
        return

      # Get the image

      image = toDraw[ int(row) ][ int(col) ]

      if image is None:
        return

      # Get bounds of visible image
      #
      # Bounds are [cx-offset,cx+offset] x [cy-offset,cy+offset]
      
      height = scale * image.shape[0]
      width  = scale * image.shape[1]

      cx     = 0.5 - translate[0]/width
      cy     = 0.5 - translate[1]/height
      offset = 0.5 / zoom

      # Find pixel position within the image array

      xFraction = (col-math.floor(col)) / (maxWidth /float(maxWidth +horizSpacing))
      yFraction = (row-math.floor(row)) / (maxHeight/float(maxHeight+vertSpacing ))

      pixelX = int( image.shape[1] * ((1-xFraction)*(cx-offset) + xFraction*(cx+offset)) )
      pixelY = int( image.shape[0] * ((1-yFraction)*(cy+offset) + yFraction*(cy-offset)) )
      
      # Perform the operation
      #
      # No operation is implemented here, but could be (e.g. image modification at the mouse position)

      # applyOperation( image, pixelX, pixelY, isFT )  

      print 'click at', pixelX, pixelY, '=', image[pixelY][pixelX]

      # Done

      glutPostRedisplay()



# Handle mouse dragging
#
# Zoom out/in with right button dragging up/down.
# Translate with left button dragging.


def mouseMotion( x, y ):

  global zoom, translate

  if currentButton == GLUT_RIGHT_BUTTON:

    # zoom

    factor = 1 # controls the zoom rate
    
    if y > initY: # zoom in
      zoom = initZoom * (1 + factor*(y-initY)/float(windowHeight))
    else: # zoom out
      zoom = initZoom / (1 + factor*(initY-y)/float(windowHeight))

  elif currentButton == GLUT_LEFT_BUTTON:

    # translate

    translate = ( initTranslate[0] + (x-initX)/zoom, initTranslate[1] + (initY-y)/zoom )

  glutPostRedisplay()


# For an image coordinate, if it's < 0 or >= max, wrap the coorindate
# around so that it's in the range [0,max-1].  This is useful dealing
# with FT images.

def wrap( val, max ):

  if val < 0:
    return val+max
  elif val >= max:
    return val-max
  else:
    return val



def forwardFT(image):
  return np.fft.fft2(image)


# Do an inverse FT
# Input is a 2D numpy array of complex values.
# Output is the same.
def inverseFT(image):
  return np.fft.ifft2(image)


# Load initial data
#
# The command line (stored in sys.argv) could have:
#
#     main.py {image filename}

if len(sys.argv) > 1:
  imageFilename = sys.argv[1]
  imagePath = os.path.join( imageDir,  imageFilename  )

image  = loadImage(  imagePath  )


# If commands exist on the command line (i.e. there are more than two
# arguments), process each command, then exit.  Otherwise, go into
# interactive mode.

if len(sys.argv) > 2:

  outputMagnitudes = True

  # process commands

  cmds = sys.argv[2:]

  while len(cmds) > 0:
    cmd = cmds.pop(0)
    if cmd == 'c':
      edgePixels = compute()
      # print 'Edge pixels:'
      # for px in edgePixels:
      #   print ' %.1f,%.1f' % (px[0],px[1])
    elif cmd[0] == 'o': # image name follows in 'cmds'
      filename = cmds.pop(0)
      allImages = [ image, smoothImage, gradientMags, gradientDirs, maximaImage, thresholdImage, edgeImage ]
      outputImage( allImages[currentImage], filename )
    elif cmd[0] in ['0','1','2','3','4','5','6']:
      currentImage = int(cmd[0]) - int('0')
    else:
      print """command '%s' not understood.
command-line arguments:
  c   - apply Canny 
  0-6 - set current image
  o   - output current image
""" % cmd

else:
      
  # Run OpenGL

  glutInit()
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB )
  glutInitWindowSize( windowWidth, windowHeight )
  glutInitWindowPosition( 50, 50 )

  glutCreateWindow( 'Canny edges' )

  glutDisplayFunc( display )
  glutKeyboardFunc( keyboard )
  glutSpecialFunc( special )
  glutReshapeFunc( reshape )
  glutMouseFunc( mouse )
  glutMotionFunc( mouseMotion )

  glDisable( GL_DEPTH_TEST )

  glutMainLoop()
