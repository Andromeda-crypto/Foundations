# visualize_experiments.py

import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def compare_metric(
    dataframes,
    labels,
    x_col,
    y_col,
    title,
    ylabel
):
    plt.figure(figsize=(12, 7))

    for df, label in zip(dataframes, labels):
        plt.plot(df[x_col], df[y_col], label=label)

    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()


def main():
    base_path = r"C:\Users\Dr.Subhash Kumar\Documents\Algorithms"

    nearest = load_csv(f"{base_path}\\nearest.csv")
    info_gain = load_csv(f"{base_path}\\info_gain.csv")
    info_gain_revisit = load_csv(
        f"{base_path}\\info_gain_revisit.csv"
    )
    uncertainty_aware = load_csv(
        f"{base_path}\\uncertainty_aware.csv"
    )

    dataframes = [
        nearest,
        info_gain,
        info_gain_revisit,
        uncertainty_aware
    ]

    labels = [
        "Nearest",
        "Info Gain",
        "Info Gain + Revisit",
        "Uncertainty Aware"
    ]

    compare_metric(
        dataframes,
        labels,
        "step",
        "mapped_percent",
        "Mapped Area Over Time",
        "Mapped Area (%)"
    )

    compare_metric(
        dataframes,
        labels,
        "step",
        "distance_traveled",
        "Distance Traveled Over Time",
        "Distance"
    )

    compare_metric(
        dataframes,
        labels,
        "step",
        "uncertainty",
        "Localization Uncertainty Over Time",
        "Uncertainty"
    )

    compare_metric(
        dataframes,
        labels,
        "step",
        "ess",
        "Particle Filter ESS Over Time",
        "ESS"
    )

    compare_metric(
        dataframes,
        labels,
        "distance_traveled",
        "mapped_percent",
        "Coverage Efficiency",
        "Mapped Area (%)"
    )


if __name__ == "__main__":
    main() 