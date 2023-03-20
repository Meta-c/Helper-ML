# plot validation and training curves seperately 

def plot_loss_curves(history):
    loss=history.history["loss"]
    val_loss=history.history["val_loss"]
    
    accuracy=history.history["accuracy"]
    val_accuracy=history.history["val_accuracy"]
    
    epochs= range(len(history.history["loss"]))
    
    # Plot loss
    plt.plot(epochs,loss,label="training_loss")
    plt.plot(epochs,val_loss,label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()
    
    
     # Plot accuracy
    plt.figure()
    plt.plot(epochs,accuracy,label="training_accuracy")
    plt.plot(epochs,val_accuracy,label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    
    
    
    
    
def view_random_image(target_dir, target_class):
    # Setup the target directory
    target_folder=target_dir+target_class
    
    # Get a random image path
    random_image=random.sample(os.listdir(target_folder),1)
    
    
    # Read the image and plot
    img=mpimg.imread(target_folder+"/"+random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    
    print(f"Image shape: {img.shape}")
    
    # return img

class_names=[]

def pred_and_plot(model,filename,image_shape,class_names=class_names):
    img=load_and_prep_image(filename,img_shape=image_shape)
    
    pred=model.predict(tf.expand_dims(img,axis=0))
    
    pred_class=class_names[np.argmax(pred)]
    
    plt.imshow(img)
    plt.title(f"Prediction : {pred_class} ")
    plt.axis(False)
    
    
def load_and_prep_image(filename, img_shape):
    img=tf.io.read_file(filename)
    img=tf.image.decode_image(img)
    img=tf.image.resize(img,size=[img_shape,img_shape])
    img=img/255.
    return img    


def unzip_file(link,file_name):
    wget.download(link)
    zip_ref=zipfile.ZipFile(file_name)
    zip_ref.extractall()
    zip_ref.close()
    
import datetime 

def create_tensorboard_callback(dir_name,experiment_name):
    log_dir=dir_name+"/"+experiment_name+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving Tensorboard log files to: {log_dir}")
    return tensorboard_callback  



# Create function for comparing training histories

def compare_histories(original_history,new_history,initial_epochs=5):
    # Get Original history measurments
    acc =original_history.history["accuracy"]
    loss=original_history.history["loss"]
    
    val_acc=original_history.history["val_accuracy"]
    val_loss=original_history.history["val_loss"]
    
    # Combine original history 
    
    total_acc = acc + new_history.history["accuracy"]
    total_loss= loss + new_history.history["loss"]
    
    total_val_acc=val_acc+ new_history.history["val_accuracy"]
    total_val_loss=val_loss+ new_history.history["val_loss"]
    
    # Make plots
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(total_acc,label="Training Accuracy")
    plt.plot(total_val_acc,label="Val Accuracy")
    plt.plot([initial_epochs-1,initial_epochs-1],plt.ylim(),label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    
    
    
    
    # Make plots loss
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,2)
    plt.plot(total_loss,label="Training Loss")
    plt.plot(total_val_loss,label="Val Loss")
    plt.plot([initial_epochs-1,initial_epochs-1],plt.ylim(),label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")  
   
   
    
# Create a ModelCheckpoint callback that saves the model's  weights
def checkpoint_callback_fun(checkpoint_path,monitor,save_weights_only,save_best_only):    
    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor=monitor,                                  
    save_weights_only=save_weights_only,
    save_best_only=save_best_only,
    save_freq="epoch", # Save every epoch,
    verbose=1)    