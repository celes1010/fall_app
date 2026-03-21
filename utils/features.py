import numpy as np

def landmarks_to_feature_vector(landmarks):
    """
    Convert a list of 33 Mediapipe pose landmarks into a 1D feature vector.

    Each landmark: (x, y, z, visibility) -> 4 values
    So total length = 33 * 4 = 132
    """
    if landmarks is None:
        return np.zeros(33 * 4, dtype=np.float32)

    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(coords, dtype=np.float32)
