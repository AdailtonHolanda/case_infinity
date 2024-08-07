import argparse
import yaml
from pydantic import BaseModel, ValidationError, validator
import cv2
from scipy.spatial.distance import cosine
import numpy as np

# Valida os tipos das entradas fornecidas no arquivo yaml.
class Config(BaseModel):
    image_a: str
    image_b: str
    output_location: str
    threshold: float

    @validator('image_a', 'image_b', 'output_location')
    def check_strings(cls, v):
        if not isinstance(v, str):
            raise ValueError(f'{v} is not a string')
        return v

    @validator('threshold')
    def check_float(cls, v):
        if not isinstance(v, float):
            raise ValueError(f'{v} is not a float')
        return v

# Lê o arquivo yaml
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Lê as imagens utilizando o caminho fornecido no arquivo yaml.
def load_images(image_a_path, image_b_path):
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    if image_a is None or image_b is None:
        raise FileNotFoundError('One or both images could not be loaded')
    return image_a, image_b

# Converte as imagens de bgr para tons de cinza.
def bgr_to_gray(image_a, image_b):
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    return gray_a, gray_b

# Redimensiona as imagens para o tamanhos de 256x256
def resize_image(gray_a, gray_b):
    resized_a = cv2.resize(gray_a, (256, 256))
    resized_b = cv2.resize(gray_b, (256, 256))
    return resized_a, resized_b

# Calcula o histograma das imagens.
def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    return hist

# Calcuna a distância cosseno.
def calculate_angular_distance(hist_a, hist_b):
    cosine_similarity = 1 - cosine(hist_a, hist_b)
    angular_distance = np.arccos(cosine_similarity) / np.pi
    return angular_distance

# Concatena as duas imagens
def concatenate_images(image_a, image_b):
    return np.concatenate((image_a, image_b), axis=1)

def save_images(image_a, image_b, path_a, path_b):
    cv2.imwrite(path_a, image_a)
    cv2.imwrite(path_b, image_b)