import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
from pathlib import Path

def compute_loss_graph(supp_loss, query_before_loss, query_after_loss, client_type):
    project_root = Path(__file__).resolve().parent.parent

    if(client_type == "garden"):
        filepath = project_root / "results/garden_loss.png"
    elif(client_type == "dining"):
        filepath = project_root / "results/dining_loss.png"
    elif(client_type == "coral"):
        filepath = project_root / "results/coral_loss.png"
    elif(client_type == "nano"):
        filepath = project_root / "results/nano_loss.png"
    else:
        print("Client type unknown, cannot plot graph")
        return

    plt.figure()
    plt.plot(supp_loss, label="Supp Loss")
    plt.plot(query_before_loss, label="Query Before Adapt Loss")
    plt.plot(query_after_loss, label="Query After Adapt Loss")
    plt.title(f"{client_type.capitalize()} Dataset Training Loss")
    plt.xlabel("Global rounds")
    plt.ylabel("Average Loss")
    #plt.ylim(0, 1)
    plt.ylim(bottom=0)
    #plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, pos: "" if y == 0 else f"{y:g}")
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def compute_acc_graph(supp_acc, query_before_acc, query_after_acc, client_type):
    project_root = Path(__file__).resolve().parent.parent

    if(client_type == "garden"):
        filepath = project_root / "results/garden_acc.png"
    elif(client_type == "dining"):
        filepath = project_root / "results/dining_acc.png"
    elif(client_type == "coral"):
        filepath = project_root / "results/coral_acc.png"
    elif(client_type == "nano"):
        filepath = project_root / "results/nano_acc.png"
    else:
        print("Client type unknown, cannot plot graph")
        return

    plt.figure()
    plt.plot(supp_acc, label="Supp Acc")
    plt.plot(query_before_acc, label="Query Before Adapt Acc")
    plt.plot(query_after_acc, label="Query After Adapt Acc")
    plt.title(f"{client_type.capitalize()} Dataset Training Accuracy")
    plt.xlabel("Global rounds")
    plt.ylabel("Average Accuracy")
    plt.ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def compute_supp_loss_graph(clients_supp_loss):
    project_root = Path(__file__).resolve().parent.parent
    filepath = project_root / "results/client_supp_losses.png"
    plt.figure()
    for client_name, supp_loss in clients_supp_loss.items():
        plt.plot(supp_loss, label=client_name)

    plt.title("Client Support Losses")
    plt.xlabel("Global rounds")
    plt.ylabel("Average Support Loss")
    plt.ylim(bottom=0)
    #plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, pos: "" if y == 0 else f"{y:g}")
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def compute_query_before_loss_graph(clients_query_before_loss):
    project_root = Path(__file__).resolve().parent.parent
    filepath = project_root / "results/client_query_before_losses.png"
    plt.figure()
    for client_name, query_loss in clients_query_before_loss.items():
        plt.plot(query_loss, label=client_name)

    plt.title("Client Query Losses Before Adaptation")
    plt.xlabel("Global rounds")
    plt.ylabel("Average Query Loss")
    plt.ylim(bottom=0)
    #plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, pos: "" if y == 0 else f"{y:g}")
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def compute_query_after_loss_graph(clients_query_after_loss):
    project_root = Path(__file__).resolve().parent.parent
    filepath = project_root / "results/client_query_after_losses.png"
    plt.figure()
    for client_name, query_loss in clients_query_after_loss.items():
        plt.plot(query_loss, label=client_name)

    plt.title("Client Query Losses After Adaptation")
    plt.xlabel("Global rounds")
    plt.ylabel("Average Query Loss")
    plt.ylim(bottom=0)
    #plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, pos: "" if y == 0 else f"{y:g}")
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def compute_supp_acc_graph(clients_supp_acc):
    project_root = Path(__file__).resolve().parent.parent
    filepath = project_root / "results/client_supp_accuracies.png"
    plt.figure()
    for client_name, supp_acc in clients_supp_acc.items():
        plt.plot(supp_acc, label=client_name)

    plt.title("Client Support Accuracies")
    plt.xlabel("Global rounds")
    plt.ylabel("Average Support Accuracy")
    plt.ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def compute_query_before_acc_graph(clients_query_before_acc):
    project_root = Path(__file__).resolve().parent.parent
    filepath = project_root / "results/client_query_before_accuracies.png"
    plt.figure()
    for client_name, query_acc in clients_query_before_acc.items():
        plt.plot(query_acc, label=client_name)

    plt.title("Client Query Accuracies Before Adaptation")
    plt.xlabel("Global rounds")
    plt.ylabel("Average Query Accuracy")
    plt.ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def compute_query_after_acc_graph(clients_query_after_acc):
    project_root = Path(__file__).resolve().parent.parent
    filepath = project_root / "results/client_query_after_accuracies.png"
    plt.figure()
    for client_name, query_acc in clients_query_after_acc.items():
        plt.plot(query_acc, label=client_name)

    plt.title("Client Query Accuracies After Adaptation")
    plt.xlabel("Global rounds")
    plt.ylabel("Average Query Accuracy")
    plt.ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()