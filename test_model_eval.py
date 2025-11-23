import numpy as np
import functions

def main():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    preds = {
        "A": np.array([0, 1, 2, 0, 0, 2, 0, 1]),  # some mistakes
        "B": np.array([0, 1, 2, 0, 1, 2, 0, 1]),  # perfect except one
    }
    accs = functions.evaluate_models(preds, y_true)
    print("Accuracies:", accs)

    best = functions.find_best_model(accs)
    print("Best model:", best)

    # Optional: show all winners if there is a tie
    # best_all = functions.find_best_model(accs, return_all_max=True)
    # print("All best models:", best_all)

if __name__ == "__main__":
    main()
