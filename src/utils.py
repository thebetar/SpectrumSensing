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
    accuracy_df = pd.read_csv("../docs/results/accuracy.csv")

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
    accuracy_df.to_csv("../docs/results/accuracy.csv", index=False)
