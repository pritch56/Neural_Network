import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the images to a range of 0 to 1
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Build the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    model.fit(x_train, y_train, epochs=3)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(f"Overall Test Accuracy: {val_acc}")

    # Make predictions on test set
    predictions = model.predict(x_test)

    # Calculate accuracy for each digit
    digit_accuracies = np.zeros(10)
    digit_counts = np.zeros(10)

    for i in range(len(y_test)):
        actual = y_test[i]
        predicted = np.argmax(predictions[i])
        digit_counts[actual] += 1
        if actual == predicted:
            digit_accuracies[actual] += 1

    # Output accuracy for each digit
    for digit in range(10):
        if digit_counts[digit] > 0:
            accuracy = digit_accuracies[digit] / digit_counts[digit]
            print(f"Accuracy for digit {digit}: {accuracy:.4f}")

    # Save the model to a file
    model.save('mnist_digit_recognizer.h5')
    print("Model saved as mnist_digit_recognizer.h5")

if __name__ == "__main__":
    main()
