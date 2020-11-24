import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def ErrorRate(y,yh):
    err = (y!=yh).mean()
    return err

def MSE(y,yh):
    mse = ((y-yh)*(y-yh)).mean()
    return mse

def PlotImage(ix_start,num_img,rows,cols,images,labels,label_names):
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace = 0.5,hspace = 0.5)
    for i in range(num_img):
        plt.subplot(rows,cols,i+1)
        plt.axis('off')
        plt.imshow(images[ix_start+i],cmap = 'Greys')
        plt.title(label_names[labels[ix_start + i]])
        
def PlotLearningCurvesLoss(hist,title = ''):
    epoch = hist.epoch
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(epoch,loss,label = 'training')
    plt.plot(epoch,val_loss,label = 'validation')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(title)
    plt.show()
    
def PlotLearningCurvesAcc(hist,title = ''):
    epoch = hist.epoch
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(epoch,acc,label = 'training')
    plt.plot(epoch,val_acc,label = 'validation')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(title)
    plt.show()
    
def load_images_from_folder(folder,imageset):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            imageset.append(img)
    return imageset

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def imagereturnshrink(imageset,index,percent):
    test_image = imageset[index]
    scale_percent = percent # percent of original size
    width = int(test_image.shape[1] * scale_percent / 100)
    height = int(test_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    test_image = cv2.resize(test_image, dim, interpolation = cv2.INTER_AREA)
    test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2BGRA)
    return test_image