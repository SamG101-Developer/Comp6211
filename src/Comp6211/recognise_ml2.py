import os
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pyefd import elliptic_fourier_descriptors

POSE = mp.solutions.pose.Pose(
    min_detection_confidence=0.0, min_tracking_confidence=0.0, static_image_mode=True, model_complexity=2)


def load_training_data(training_folder: str) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []

    for file in os.listdir(training_folder):
        image = cv2.imread(os.path.join(training_folder, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features = extract_body_shape_features(image)

        x.append(features)
        y.append(file[:7])

    return np.array(x), np.array(y)


def train_model(x: np.ndarray, y: np.ndarray) -> SVC:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = SVC(kernel="rbf", C=1.0, random_state=101, probability=True, max_iter=-1)
    model.fit(x_train, y_train)
    return model


def classify(image_path: str, model: SVC, scaler: StandardScaler) -> str:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    features = extract_body_shape_features(image)
    features = scaler.transform([features])
    return model.predict(features)[0]


# sokalsneath, minkowski, euclidian, chebyshev, canberra
def calc_dist(a, b) -> float | np.floating:
    return distance.euclidean(a, b)


def extract_body_shape_features(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = POSE.process(image)
    landmarks = results.pose_landmarks.landmark

    features = []

    def get_length(l1: int, l2: int) -> float:
        return calc_dist((landmarks[l1].x, landmarks[l1].y), (landmarks[l2].x, landmarks[l2].y))

    def get_angle(l1: int, l2: int, l3: int) -> float:
        a = landmarks[l1].x, landmarks[l1].y
        b = landmarks[l2].x, landmarks[l2].y
        c = landmarks[l3].x, landmarks[l3].y
        ba = a[0] - b[0], a[1] - b[1]
        bc = c[0] - b[0], c[1] - b[1]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return angle

    # Get the distance between every pair of landmarks
    left_upper_body = get_length(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP)
    right_upper_body = get_length(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP)
    nose_to_left_heel = get_length(mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_HEEL)
    nose_to_right_heel = get_length(mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.RIGHT_HEEL)
    shoulder_to_shoulder = get_length(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
    hip_to_hip = get_length(mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP)
    ear_to_ear = get_length(mp.solutions.pose.PoseLandmark.LEFT_EAR, mp.solutions.pose.PoseLandmark.RIGHT_EAR)

    features.extend([
        nose_to_left_heel, nose_to_right_heel, left_upper_body, right_upper_body,
        shoulder_to_shoulder, hip_to_hip, ear_to_ear
    ])

    # Get angles (should/hip/knee and shoulder/shoulder/hip)
    left_shoulder_hip_knee = get_angle(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE)
    right_shoulder_hip_knee = get_angle(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE)
    left_shoulder_shoulder_hip = get_angle(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP)
    right_shoulder_shoulder_hip = get_angle(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP)

    features.extend([
        left_shoulder_hip_knee, right_shoulder_hip_knee, left_shoulder_shoulder_hip, right_shoulder_shoulder_hip,
    ])

    # Remove the background.
    copy_image = image.copy()
    lab = cv2.cvtColor(copy_image, cv2.COLOR_RGB2LAB)
    a_channel = lab[:, :, 1]
    th = cv2.threshold(a_channel, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    masked = cv2.bitwise_and(copy_image, copy_image, mask=th)
    masked[th == 0] = (255, 255, 255)

    # Crop the image to only include the person.
    height, width, _ = masked.shape
    masked = masked[int(height * 0.2):int(height * 0.95), int(width * 0.35):int(width * 0.65)]
    height, width, _ = masked.shape

    # Crop the image, based on the mediapipe outputs, to only include the head of the person.
    masked = masked[:int(landmarks[mp.solutions.pose.PoseLandmark.NOSE].y * height * 0.75), :]

    # Add a white bar along the top and bottom of the image.
    white_bar_1 = np.ones((int(height * 0.05), width, 3), np.uint8) * 255
    white_bar_2 = np.ones((int(height * 0.05), width, 3), np.uint8) * 255
    masked = np.vstack((white_bar_1, masked, white_bar_2))
    height, width, _ = masked.shape

    # Add a white bar along the left and right of the image.
    white_bar_3 = np.ones((height, int(width * 0.05), 3), np.uint8) * 255
    white_bar_4 = np.ones((height, int(width * 0.05), 3), np.uint8) * 255
    masked = np.hstack((white_bar_3, masked, white_bar_4))
    height, width, _ = masked.shape

    # Moments
    grey_image = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grey_image, 150, 255, 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    head_contour = sorted(contours, key=cv2.contourArea, reverse=True)[1]
    moments = cv2.moments(head_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    features.extend(hu_moments)

    # Elliptic Fourier Descriptors
    coeffs = elliptic_fourier_descriptors(np.squeeze(head_contour), order=10, normalize=True)
    features.extend(coeffs.flatten()[3:])

    return np.array(features, dtype=np.float64)


def calculate_intra_and_inter_distances(subjects: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    collapsed = defaultdict(list)
    intra_distances = []
    inter_distances = []

    # Collapse common subjects
    for subject, feature_group in zip(subjects, features):
        collapsed[subject].append(feature_group)

    # Intra
    for subject, feature_group in collapsed.items():
        d = calc_dist(feature_group[0], feature_group[1])
        intra_distances.append(d)

    # Inter
    for i, (subject1, feature_group1) in enumerate(collapsed.items()):
        for j, (subject2, feature_group2) in enumerate(collapsed.items()):
            if i >= j: continue
            ff1 = np.mean(feature_group1, axis=0)
            ff2 = np.mean(feature_group2, axis=0)
            d = calc_dist(ff1, ff2)
            inter_distances.append(d)

    # Convert to numpy arrays
    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)
    return intra_distances, inter_distances


def plot_histograms(intra: np.ndarray, inter: np.ndarray) -> None:
    plt.figure(figsize=(10, 10))
    plt.hist(intra.tolist(), bins=50, alpha=0.5, color="b", label="Intra-class distances", density=True)
    plt.hist(inter.tolist(), bins=50, alpha=0.5, color="r", label="Inter-class distances", density=True)

    plt.xlabel("Euclidean distance")
    plt.ylabel("Frequency density")
    plt.title("Histogram of intra-class distances")
    plt.legend()
    plt.show()


def main():

    TRUE_MATCHES = {
        "DSC00165.JPG": "021z001",
        "DSC00166.JPG": "021z001",
        "DSC00167.JPG": "021z002",
        "DSC00168.JPG": "021z002",
        "DSC00169.JPG": "021z003",
        "DSC00170.JPG": "021z003",
        "DSC00171.JPG": "021z004",
        "DSC00172.JPG": "021z004",
        "DSC00173.JPG": "021z005",
        "DSC00174.JPG": "021z005",
        "DSC00175.JPG": "021z006",
        "DSC00176.JPG": "021z006",
        "DSC00177.JPG": "021z007",
        "DSC00178.JPG": "021z007",
        "DSC00179.JPG": "021z008",
        "DSC00180.JPG": "021z008",
        "DSC00181.JPG": "021z009",
        "DSC00182.JPG": "021z009",
        "DSC00183.JPG": "021z010",
        "DSC00184.JPG": "021z010",
        "DSC00185.JPG": "024z011",
        "DSC00186.JPG": "024z011",
    }

    x_train, y_train = load_training_data("./res/biometrics/training")
    svm_model = train_model(x_train, y_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    intra, inter = calculate_intra_and_inter_distances(y_train, x_train)
    plot_histograms(intra, inter)

    correct_counter = 0
    for file, true_match in TRUE_MATCHES.items():
        predicted_match = classify(f"./res/biometrics/test/{file}", svm_model, scaler)
        correct_counter += (correct := predicted_match == true_match)
        print(f"[{file} | Predicted: {predicted_match}, True: {true_match}. Match? {correct}]")

    print(f"CCR: {correct_counter / len(TRUE_MATCHES)}")
