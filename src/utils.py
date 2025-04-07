import torch
import netron
import matplotlib.pyplot as plt


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
