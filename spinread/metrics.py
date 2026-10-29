import numpy as np

from spinread.physics import SPIN_COMPONENT_NAMES, TOP


def r_squared(true_values, predicted_values):
    true_values, predicted_values = np.asarray(true_values), np.asarray(
        predicted_values
    )
    sum_of_squared_errors = np.sum((true_values - predicted_values) ** 2)
    sum_of_squares_around_mean = np.sum((true_values - true_values.mean()) ** 2)
    return float(1 - sum_of_squared_errors / sum_of_squares_around_mean)


def root_mean_squared_error(true_values, predicted_values):
    return float(
        np.sqrt(np.mean((np.asarray(true_values) - np.asarray(predicted_values)) ** 2))
    )


def sign_accuracy(true_values, predicted_values):
    return float(np.mean(np.sign(true_values) == np.sign(predicted_values)))


def score(true_spins, predicted_spins):
    # R^2 and RMSE (rev/s) of topspin and sidespin, plus the topspin sign accuracy, as one flat row
    true_spins, predicted_spins = np.asarray(true_spins), np.asarray(predicted_spins)
    scores = {}
    for component_number, component_name in enumerate(SPIN_COMPONENT_NAMES):
        scores[f"r2_{component_name}"] = r_squared(
            true_spins[:, component_number], predicted_spins[:, component_number]
        )
        scores[f"rmse_{component_name}"] = root_mean_squared_error(
            true_spins[:, component_number], predicted_spins[:, component_number]
        )
    scores["sign_top"] = sign_accuracy(true_spins[:, TOP], predicted_spins[:, TOP])
    scores["how_many"] = int(len(true_spins))
    return scores
