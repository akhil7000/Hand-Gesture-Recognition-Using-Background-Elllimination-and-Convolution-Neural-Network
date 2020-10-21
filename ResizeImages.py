from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((89,100), Image.ANTIALIAS)
    print(img.size)
    img.save(imageName)

for i in range(0, 100):
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("Dataset/OpenEyesImages/Train/open_" + str(i) + '.png')
    resizeImage("Dataset/OpenEyesImages/Test/open_" + str(i) + '.png')
    resizeImage("Dataset/ClosedEyesImages/Train/closed_" + str(i) + '.png')
    resizeImage("Dataset/ClosedEyesImages/Test/closed_" + str(i) + '.png')


