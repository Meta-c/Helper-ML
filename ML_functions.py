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
    
    
#########################################################################   


    
    
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




#########################################################################




def pred_and_plot(model,filename,image_shape,class_names=class_names):
    img=load_and_prep_image(filename,img_shape=image_shape)
    
    pred=model.predict(tf.expand_dims(img,axis=0))
    
    pred_class=class_names[np.argmax(pred)]
    
    plt.imshow(img)
    plt.title(f"Prediction : {pred_class} ")
    plt.axis(False)
    
    
    
    
    
########################################################################    
    
    
    
    
    
    
def load_and_prep_image(filename, img_shape,scale=True):
    img=tf.io.read_file(filename)
    img=tf.image.decode_image(img,channels=3)
    img=tf.image.resize(img,size=[img_shape,img_shape])
    if scale:
        return img/255.
    else:
        return img
       



#########################################################################





def unzip_file(link,file_name):
    wget.download(link)
    zip_ref=zipfile.ZipFile(file_name)
    zip_ref.extractall()
    zip_ref.close()
    
import datetime 



#########################################################################




def create_tensorboard_callback(dir_name,experiment_name):
    log_dir=dir_name+"/"+experiment_name+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving Tensorboard log files to: {log_dir}")
    return tensorboard_callback  




#########################################################################





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
   
   



#########################################################################

    
# Create a ModelCheckpoint callback that saves the model's  weights
def checkpoint_callback_fun(checkpoint_path,monitor,save_weights_only,save_best_only):    
    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor=monitor,                                  
    save_weights_only=save_weights_only,
    save_best_only=save_best_only,
    save_freq="epoch", # Save every epoch,
    verbose=1)    
    


#########################################################################

    
def confusion_matrix(y_true,y_pred,classes,figsize):

    num_classes=len(classes)
    
    # Create the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # Populate the confusion matrix
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1

    # Define the figure and axis
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    
    
    # Create a heatmap of the confusion matrix
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # Add a colorbar to the figure
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set the axis labels and ticks
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(np.array(classes), fontsize=8)
    ax.set_yticklabels(np.array(classes), fontsize=8)
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)

    # Rotate the x-axis labels for better visibility
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Add the confusion values to the heatmap
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color="w", fontsize=6)

    # Add a title to the figure
    ax.set_title("Confusion matrix", fontsize=20)

    # Display the figure
    plt.show()



#########################################################################


# MASE implementation
def mean_absolute_scaled_error(y_true,y_pred):
    mae=tf.reduce_mean(tf.abs(y_true-y_pred))
    
    # Find MAE of naive forecast (no seasonality)
    mae_naive_no_season= tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    
    return mae/mae_naive_no_season



#########################################################################


# Create function to take model preds and truth values \
def evaluate_preds(y_true,y_pred):
    # Make sure float32 dtype 
    y_true= tf.cast(y_true,dtype=tf.float32)
    y_pred=tf.cast(y_pred,dtype=tf.float32)
    
    # Calculate various evaluation metrics 
    mae = tf.keras.metrics.mean_absolute_error(y_true,y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true,y_pred)
    rmse= tf.sqrt(mse)
    mape=tf.keras.metrics.mean_absolute_percentage_error(y_true,y_pred)
    mase = mean_absolute_scaled_error(y_true,y_pred)
    
    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse":rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy(),
            }


#########################################################################
