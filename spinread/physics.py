import numpy as np

SAMPLES_PER_SECOND = 500  # the simulator saves a position every 2 ms
GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81])
BALL_RADIUS = 0.02  # m
TABLE_HEIGHT = 0.76  # m, top of the simulator's table; the ball centre sits at 0.78 m when it touches
MAGNUS_COEFFICIENT, DRAG_COEFFICIENT = (
    0.0075,
    0.133,
)  # fitted on a sample of the released data
# the two components of a spin label: topspin/backspin (about y~) and sidespin (about z~).
# gyrospin, about the flight direction x~, produces no Magnus force at launch and is left
# to the final report, which adds the bounce that reveals it
SPIN_COMPONENT_NAMES = ["top", "side"]
TOP = SPIN_COMPONENT_NAMES.index("top")
LABEL_ROWS = [1, 2]  # the y~ and z~ rows of ball_frame()


def fit_polynomial_per_axis(sample_times, values_per_sample, degree, reference_time):
    design_matrix = np.vander(
        sample_times - reference_time, degree + 1, increasing=True
    )
    coefficients, *_ = np.linalg.lstsq(design_matrix, values_per_sample, rcond=None)
    return coefficients, design_matrix


def ball_frame(velocity):
    along_flight = np.array([velocity[0], velocity[1], 0.0])
    along_flight /= np.linalg.norm(along_flight)
    up = np.array([0.0, 0.0, 1.0])
    sideways = np.cross(up, along_flight)
    return np.stack([along_flight, sideways, up])


def spin_label(launch_spin_world, launch_velocity):
    # (topspin, sidespin) in rev/s, in the ball frame of the true launch velocity
    return (ball_frame(launch_velocity) @ np.asarray(launch_spin_world) / (2 * np.pi))[
        LABEL_ROWS
    ]


def air_acceleration(velocity, spin):
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    return -DRAG_COEFFICIENT * speed * velocity + MAGNUS_COEFFICIENT * np.cross(
        spin, velocity
    )


def first_contact_sample(sample_times, positions, bounce_times):
    if len(bounce_times) == 0:
        return len(sample_times)
    heights = positions[:, 2]
    guessed_sample = int(
        round((bounce_times[0] - sample_times[0]) * SAMPLES_PER_SECOND)
    )
    search_start, search_end = max(0, guessed_sample - 25), min(
        len(heights), guessed_sample + 25
    )
    contact_sample = search_start + int(np.argmin(heights[search_start:search_end]))
    while (
        contact_sample > 0
        and heights[contact_sample - 1] < TABLE_HEIGHT + BALL_RADIUS + 0.001
    ):
        contact_sample -= 1
    return contact_sample


def free_flight_end(sample_times, positions, bounce_times):
    if len(bounce_times) == 0:
        return len(sample_times)
    return max(0, first_contact_sample(sample_times, positions, bounce_times) - 2)
