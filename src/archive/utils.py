import time
import torch
import netron
import matplotlib.pyplot as plt
import pandas as pd


def show_graph_loss(
    loss_history, num_epochs, plot_label="Convolution", label="avg", filename="loss"
):
    # Evaluate the model
    plt.plot(loss_history)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{plot_label} training Loss ({label})")

    plt.xlim(0, num_epochs)
    plt.ylim(0, max(1, max(loss_history)))

    last_x = len(loss_history) - 1
    last_y = loss_history[-1]
    plt.scatter(last_x, last_y, color="red")
    plt.text(last_x - 1, last_y + 0.02, f"{last_y:.4f}", verticalalignment="bottom")

    plt.savefig(filename)
    plt.show()


def show_graph_model(model, test_data, device, filename="model"):
    # Example model and input
    x = torch.FloatTensor(test_data).to(device).unsqueeze(0)

    # Generate and save the graph
    torch.onnx.export(
        model, x, filename + ".onnx", input_names=["input"], output_names=["output"]
    )

    # Convert the ONNX model to PNG
    netron.start()


def log_accuracy(
    model,
    label,
    data,
    accuracy,
    training_time,
    num_epochs,
    avg_response_time,
    avg_loss,
    min_loss,
    max_loss,
    sampling_rate=1,
):
    accuracy_df = pd.read_csv("../../docs/results/accuracy.csv")

    accuracy_df = pd.concat(
        [
            accuracy_df,
            pd.DataFrame(
                [
                    {
                        "model": model,
                        "label": label,
                        "accuracy": accuracy * 100,
                        "training_time": training_time,
                        "epochs": num_epochs,
                        "avg_response_time": avg_response_time,
                        "avg_loss": avg_loss,
                        "min_loss": min_loss,
                        "max_loss": max_loss,
                        "sequence_length": data.shape[1],
                        "sampling_rate": sampling_rate,
                    }
                ]
            ),
        ]
    )
    accuracy_df.to_csv("../../docs/results/accuracy.csv", index=False)


def run_tests(model, criterion, X_test, y_test, batch_size, device):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        response_times = []

        for i in range(0, len(X_test), batch_size):
            X_batch = torch.FloatTensor(X_test[i : i + batch_size]).to(device)
            y_batch = torch.FloatTensor(y_test[i : i + batch_size]).to(device)

            time_start = time.time()
            outputs = model(X_batch)
            time_end = time.time()

            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

            # Convert probabilities to binary predictions
            predicted = (outputs > 0).float()

            # Calculate the number of correct predictions
            correct_predictions += (predicted == y_batch).sum().item()
            total_samples += y_batch.numel()
            response_times.append(time_end - time_start)

        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(X_test)
        avg_response_time = sum(response_times) / len(response_times)

    print(f"Average testing loss: {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.4f}%")

    return {
        "total_loss": total_loss,
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "avg_response_time": avg_response_time,
    }
