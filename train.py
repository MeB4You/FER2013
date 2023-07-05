import time
from utils import (get_model, plot_acc_loss, show_model_architectures, compile_model, get_callbacks)
from load_images import (get_dataset, data_augmentation, show_augmented_images,show_images)

visualization = True
show_architecture = False
seed = 1234

def run_model():
    labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

    X_train, y_train, X_val, y_val, X_test, y_test = get_dataset()
    datagen = data_augmentation(X_train)
    model = get_model(len(labels))
    print("--------------------------------------------------------")
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape : " + str(X_test.shape))
    print("Y_test shape : " + str(y_test.shape))
    print("X_val shape  : " + str(X_val.shape))
    print("Y_val shape  : " + str(y_val.shape))
    print("--------------------------------------------------------")
    if visualization:
        show_images(X_train,y_train)
        show_augmented_images(datagen, X_train, y_train)
        if show_architecture:
            show_model_architectures(model)

    BS = 128
    num_epochs = 400
    train_generator = datagen.flow(X_train, y_train, batch_size=BS,shuffle = True,seed = seed)
    compile_model(model)

    start_time = time.strftime("%H:%M:%S", time.localtime())
    s_t = time.time()
    history = model.fit(x = train_generator, steps_per_epoch = X_train.shape[0] // BS,
                    epochs=num_epochs,verbose=1, validation_data=(X_val, y_val),callbacks = get_callbacks() ,shuffle = True)
    
    end_time = time.strftime("%H:%M:%S", time.localtime())
    e_t = time.time()
    print("--------------------------------------------------------")
    print(f"Starting time: {start_time}")
    print(f"Ending time  : {end_time}")
    print("Total Training time:", time.strftime("%H:%M:%S",time.gmtime( e_t-s_t)))
    print("--------------------------------------------------------")

    plot_acc_loss(history, num_epochs)

run_model()