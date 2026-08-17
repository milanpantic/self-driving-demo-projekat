import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


DEFAULT_CSV_PATH = "rezultati_predikcije.csv"
DEFAULT_OUTPUT_PATH = "rezultati_predikcije_grafik.png"
DEFAULT_FPS = 14.5


def create_prediction_plot(
    csv_path=DEFAULT_CSV_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    fps=DEFAULT_FPS,
):
    df = pd.read_csv(csv_path)
    actual = df["actual_steering_angle"].to_numpy(dtype=float)
    predicted = df["predicted_steering_angle"].to_numpy(dtype=float)
    time_seconds = np.arange(len(df)) / fps

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        time_seconds,
        actual,
        label="Stvarni ugao",
        color="#1565c0",
        linewidth=1.5,
    )
    axis.plot(
        time_seconds,
        predicted,
        label="Predviđeni ugao",
        color="#ef6c00",
        linewidth=1.5,
        alpha=0.9,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Stvarni i predviđeni ugao kroz vreme")
    axis.set_xlabel("Vreme videa (s)")
    axis.set_ylabel("Ugao: levo (−) / desno (+)")
    axis.xaxis.set_major_locator(MultipleLocator(1))
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    created_path = create_prediction_plot()
    print(f"Grafikon sačuvan: {created_path}")
