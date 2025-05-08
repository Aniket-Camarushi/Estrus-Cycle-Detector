# my_training.py

"""
Enhanced training script for Secreit model.
- Splits large images
- Trains model (last layer & full fine-tuning)
- Saves model weights
- Logs & plots training history
"""

import os
os.environ["OMP_NUM_THREADS"] = "14"  # For OpenMP threads
os.environ["TF_NUM_INTRAOP_THREADS"] = "14"  # For internal parallelism (operations)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"  # For inter-operations parallelism

import glob
import json
from keras.applications import VGG16
from keras import models, layers, optimizers
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
import Secreit

# 1. Split images into quadrants
def split_images(dataset_dir='data', output_dir='data/split'):
    for dataset in ["train", "validation", "test"]:
        for stage in ["D", "E", "P"]:
            input_paths = glob.glob(f"./{dataset_dir}/{dataset}/{stage}/*.png")
            output_stage_dir = f"{output_dir}_{dataset}/{stage}"
            os.makedirs(output_stage_dir, exist_ok=True)

            for path in input_paths:
                img_id = os.path.basename(path)[:-4]
                img = image.load_img(path, target_size=(480, 640))
                img_array = image.img_to_array(img)

                quads = [
                    img_array[:240, :320],
                    img_array[240:480, :320],
                    img_array[:240, 320:640],
                    img_array[240:480, 320:640]
                ]

                for i, quad in enumerate(quads, 1):
                    image.save_img(f"{output_stage_dir}/{img_id}_{i}.png", quad)

# 2. Load image paths & initialize generators
def load_data():
    trainD = glob.glob("./data/split_train/D/*.png")
    trainE = glob.glob("./data/split_train/E/*.png")
    trainP = glob.glob("./data/split_train/P/*.png")

    print("Sample image shape:", image.load_img(trainD[0]).size)

    valD = glob.glob("./data/split_validation/D/*.png")
    valE = glob.glob("./data/split_validation/E/*.png")
    valP = glob.glob("./data/split_validation/P/*.png")

    testD = glob.glob("./data/split_test/D/*.png")
    testE = glob.glob("./data/split_test/E/*.png")
    testP = glob.glob("./data/split_test/P/*.png")

    # Add this debug check:
    print(f"Found {len(trainD)} D, {len(trainE)} E, {len(trainP)} P training images")

    return (
        Secreit.TrainGenerator(trainD, trainE, trainP, 8),
        Secreit.ValidationGenerator(valD, valE, valP, 4),
        Secreit.ValidationGenerator(testD, testE, testP, 4)
    )

# 3. Build the VGG16-based model
# Update the build_model() function to match Secreit's architecture:
def build_model():
    # Use Secreit's custom model instead of standard VGG16
    model = Secreit.vgg_model(
        weight_path="./data/weights.hdf5",  # Using provided weights
        input_shape=(240, 320, 3),
        classes=3
    )
    
    model.compile(
        loss='categorical_hinge',  # Changed from hinge loss
        optimizer=optimizers.Nadam(lr=1e-5),
        metrics=['accuracy']
    )
    return model

# 4. Train last layer and save history
def train_last_layer(model, train_gen, val_gen):
    cb = [
        EarlyStopping(monitor='val_loss', patience=5, 
                    min_delta=0.001, restore_best_weights=True),
        ModelCheckpoint('./data/best_last_layer.h5',
                      monitor='val_acc', mode='max', save_best_only=True),
        CSVLogger('./data/last_layer_learning/training_log.csv')
    ]
        
    # cb = ModelCheckpoint(filepath="./data/last_layer_learning/weight_epoch{epoch:02d}-{val_loss:.2f}.h5")

    # Here, use your custom generator to yield batches
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=20,                               # Changed from 50 to 20 for faster training
        callbacks=cb
    )
    return history

# 5. Fine-tune all layers
def fine_tune_all_layers(model, train_gen, test_gen):
    # Add this verification:
    print("\n=== FINE-TUNING VERIFICATION ===")
    print(f"Loading weights from: ./data/best_last_layer.h5")
    print(f"Trainable layers before unfreezing: {sum([l.trainable for l in model.layers])}")

    model.load_weights("./data/best_last_layer.h5")  # Replace with best file if needed
    print("\nLayer Trainability Before:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name} - {layer.trainable}")

    # Only unfreeze CONV layers
    for layer in model.layers:
        if 'conv' in layer.name:
            layer.trainable = True

    # print(f"Trainable layers after unfreezing: {sum([l.trainable for l in model.layers])}")

    model.compile(
        loss='categorical_hinge',
        optimizer=optimizers.Nadam(lr=1e-5),
        metrics=['accuracy']
    )

    cb = [
        EarlyStopping(monitor='val_loss', patience=5,
                     min_delta=0.001, restore_best_weights=True),
        ModelCheckpoint('./data/all_tuning/best_finetuned.h5',
                      monitor='val_acc', mode='max', save_best_only=True),
        CSVLogger('./data/all_tuning/finetuning_log.csv')
    ]

    # cb = ModelCheckpoint(filepath="./data/all_tuning/weight_epoch{epoch:02d}-{val_loss:.2f}.h5")
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=test_gen,
        validation_steps=len(test_gen),
        epochs=20,                              # Changed from 100 to 50 for faster training
        callbacks=cb
    )
    return history

# 6. Save history to JSON
def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)

# 7. Plot training curves
def plot_history(history, title_prefix="Training"):
    acc = history.history.get('acc', [])
    val_acc = history.history.get('val_acc', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    if not loss or not val_loss:
        print(f"[WARNING] Empty loss history. Skipping plot for {title_prefix}.")
        return

    if not acc or not val_acc:
        print(f"[WARNING] Accuracy history is empty. Plotting only loss for {title_prefix}.")
        epochs = range(1, len(loss) + 1)
        print("loss:", loss)
        print("val_loss:", val_loss)
        print("epochs:", list(epochs))

        plt.figure(figsize=(6, 5))
        plt.plot(epochs, loss, 'b', label='Train Loss')
        plt.plot(epochs, val_loss, 'r', label='Val Loss')
        plt.title(f'{title_prefix} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./Results/{title_prefix.lower().replace(' ', '_')}_loss_only.png")
        plt.show(block=False)
        plt.pause(30)
        plt.close()
        return

    # If everything is available, plot both
    epochs = range(1, len(acc) + 1)
    print("acc:", acc)
    print("val_acc:", val_acc)
    print("loss:", loss)
    print("val_loss:", val_loss)
    print("epochs:", list(epochs))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Train Acc')
    plt.plot(epochs, val_acc, 'r', label='Val Acc')
    plt.title(f'{title_prefix} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Val Loss')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"./Results/{title_prefix.lower().replace(' ', '_')}_metrics.png")
    plt.show(block=False)
    plt.pause(30)
    plt.close()

# 8. Main
if __name__ == "__main__":
    print("Splitting images...")
    split_images()

    print("Loading dataset...")
    train_gen, val_gen, test_gen = load_data()

    # ADD THESE DEBUG LINES RIGHT HERE:
    print("\n=== DATA VERIFICATION ===")
    print(f"Total training samples: {len(train_gen.file_paths)}")
    print(f"Total validation samples: {len(val_gen.file_paths)}")
    print(f"Training batch size: {train_gen.batch_size}")
    print(f"Validation batch size: {val_gen.batch_size}")
    print(f"Training steps per epoch: {len(train_gen)}")
    print(f"Validation steps: {len(val_gen)}\n")

    print("Building model...")
    model = build_model()

    print("Training last layer...")
    last_layer_history = train_last_layer(model, train_gen, val_gen)
    save_history(last_layer_history, './Results/last_layer_history.json')
    plot_history(last_layer_history, title_prefix="Last Layer Training")

    print("Fine-tuning all layers...")
    fine_tune_history = fine_tune_all_layers(model, train_gen, test_gen)
    save_history(fine_tune_history, './Results/fine_tune_history.json')
    plot_history(fine_tune_history, title_prefix="Full Model Fine-tuning")

    print("All done ✅")

