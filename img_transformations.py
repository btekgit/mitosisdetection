#############################################
##             TRANSFORMATIONS             ##
#############################################

import os, sys, pdb, numpy

from PIL import Image,ImageChops,ImageOps,ImageDraw




NHOMO = 8
JPG=[70,50,30]
ROTS = [3,6,9,12,15]
SCALES=[1.5**0.5,1.5,1.5**1.5,1.5**2,1.5**2.5]
#parameters computed on ILSVRC10 dataset
lcolor = [ 381688.61379382 , 4881.28307136,  2316.10313483]
pcolor = [[-0.57848371, -0.7915924,   0.19681989],
          [-0.5795621 ,  0.22908373, -0.78206676],
          [-0.57398987 , 0.56648223 , 0.59129816]]

#pre-generated gaussian values
alphas = [[0.004894 , 0.153527, -0.012182],
          [-0.058978, 0.114067, -0.061488],
          [0.002428, -0.003576, -0.125031]]

def gen_colorimetry(i):
    p1r = pcolor[0][0]
    p1g = pcolor[1][0]
    p1b = pcolor[2][0]
    p2r = pcolor[0][1]
    p2g = pcolor[1][1]
    p2b = pcolor[2][1]
    p3r = pcolor[0][2]
    p3g = pcolor[1][2]
    p3b = pcolor[2][2]

    l1 = numpy.sqrt(lcolor[0])
    l2 = numpy.sqrt(lcolor[1])
    l3 = numpy.sqrt(lcolor[2])

    if i<=3:
        alpha = alphas[i]
    else:
        numpy.random.seed(i*3)
        alpha = numpy.random.randn(3,0,0.01)
    a1 = alpha[0]
    a2 = alpha[1]
    a3 = alpha[2]

    return (a1*l1*p1r + a2*l2*p2r + a3*l3*p3r,
            a1*l1*p1g + a2*l2*p2g + a3*l3*p3g,
            a1*l1*p1b + a2*l2*p2b + a3*l3*p3b)

def gen_crop(i,w,h):
    numpy.random.seed(4*i)
    x0 = numpy.random.random()*(w/4)
    y0 = numpy.random.random()*(h/4)
    x1 = w - numpy.random.random()*(w/4)
    y1 = h - numpy.random.random()*(h/4)

    return (int(x0),int(y0),int(x1),int(y1))

def gen_homo(i,w,h):
    if i==0:
        return (0,0,int(0.125*w),h,int(0.875*w),h,w,0)
    elif i==1:
      return (0,0,int(0.25*w),h,int(0.75*w),h,w,0)
    elif i==2:
        return (0,int(0.125*h),0,int(0.875*h),w,h,w,0)
    elif i==3:
      return (0,int(0.25*h),0,int(0.75*h),w,h,w,0)
    elif i==4:
        return (int(0.125*w),0,0,h,w,h,int(0.875*w),0)
    elif i==5:
        return (int(0.25*w),0,0,h,w,h,int(0.75*w),0)
    elif i==6:
        return (0,0,0,h,w,int(0.875*h),w,int(0.125*h))
    elif i==7:
        return (0,0,0,h,w,int(0.75*h),w,int(0.25*h))
    else:
        assert False


def rot(image,angle,fname):
    white = Image.new('L',image.size,"white")
    wr = white.rotate(angle,Image.NEAREST,expand=0)
    im = image.rotate(angle,Image.BILINEAR,expand=0)
    try:
      image.paste(im,wr)
    except ValueError:
      print >>sys.stderr, 'error: image do not match '+fname
    return image

def gen_corner(n, w, h):
    x0 = 0
    x1 = w
    y0 = 0
    y1 = h
    
    rat = 256 - 227

    if n == 0: #center
        x0 = (rat*w)/(2*256.0)
        y0 = (rat*h)/(2*256.0)
        x1 = w - (rat*w)/(2*256.0)
        y1 = h - (rat*h)/(2*256.0)
    elif n == 1:
        x0 = (rat*w)/256.0
        y0 = (rat*h)/256.0
    elif n == 2:
        x1 = w - (rat*w)/256.0
        y0 = (rat*h)/256.0
    elif n == 3:
        x1 = w - (rat*w)/256.0
        y1 = h - (rat*h)/256.0
    else:
        assert n==4
        x0 = (rat*w)/256.0
        y1 = h - (rat*h)/256.0

    return (int(x0),int(y0),int(x1),int(y1))

#the main fonction to call
#takes a image input path, a transformation and an output path and does the transformation
def gen_trans(imgfile,trans,outfile):
    for trans in trans.split('*'):
        image = Image.open(imgfile)
        w,h = image.size
        if trans=="plain":
            image.save(outfile,"JPEG",quality=100)
        elif trans=="flip":
            ImageOps.mirror(image).save(outfile,"JPEG",quality=100)
        elif trans.startswith("crop"):
            c = int(trans[4:])
            image.crop(gen_crop(c,w,h)).save(outfile,"JPEG",quality=100)
        elif trans.startswith("homo"):
            c = int(trans[4:])
            image.transform((w,h),Image.QUAD,
                            gen_homo(c,w,h),
                            Image.BILINEAR).save(outfile,"JPEG",quality=100)
        elif trans.startswith("jpg"):
            image.save(outfile,quality=int(trans[3:]))
        elif trans.startswith("scale"):
            scale = SCALES[int(trans.replace("scale",""))]
            image.resize((int(w/scale),int(h/scale)),Image.BILINEAR).save(outfile,"JPEG",quality=100)
        elif trans.startswith('color'):
            (dr,dg,db) = gen_colorimetry(int(trans[5]))
            table = numpy.tile(numpy.arange(256),(3))
            table[   :256]+= dr
            table[256:512]+= dg
            table[512:   ]+= db
            image.convert("RGB").point(table).save(outfile,"JPEG",quality=100)
        elif trans.startswith('rot-'):
            angle =int(trans[4:])
            for i in range(angle):
                image = rot(image,-1,outfile)
            image.save(outfile,"JPEG",quality=100)
        elif trans.startswith('rot'):
            angle =int(trans[3:])
            for i in range(angle):
                image = rot(image,1,outfile)
            image.save(outfile,"JPEG",quality=100)
        elif trans.startswith('corner'):
            i = int(trans[6:])
            image.crop(gen_corner(i,w,h)).save(outfile,"JPEG",quality=100)
        else:
            assert False, "Unrecognized transformation: "+trans
        imgfile = outfile # in case we iterate


#Our 41 transformations used in the CVPR paper
def get_all_trans():
  transformations = (["plain","flip"]
                    +["crop%d"%i for i in range(NCROPS)] 
                    +["homo%d"%i for i in range(NHOMO)]
                    +["jpg%d"%i for i in JPG]
                    +["scale0","scale1","scale2","scale3","scale4"]
                    +["color%d"%i for i in range(3)]
                    +["rot-%d"%i for i in ROTS]
                    +["rot%d"%i for i in ROTS])
  return transformations

#transformations used at test time in deep architectures
def get_deep_trans():
    return ['corner0','corner1','corner2','corner3','corner4','corner0*flip','corner1*flip','corner2*flip','corner3*flip','corner4*flip']

if __name__=="__main__":
    img_input = sys.argv[1]
    outpath = sys.argv[2]
    if len(sys.argv)>= 4:
        trans = sys.argv[3]
        if not trans.startswith("["):
            trans = [trans]
        else:
            trans = eval(trans)
    else:
        trans = get_all_trans()
    print "Generating transformations and storing in %s"%(outpath)
    for t in trans:
        gen_trans(img_input,t,outpath+'/%s_%s.jpg'%(".".join(img_input.split("/")[-1].split(".")[:-1]),t))
    print "Finished. Transformations generated: %s"%(" ".join(trans))



























