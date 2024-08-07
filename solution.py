import argparse
import pydantic
from utils import *

def main(yaml_path):
    config_data = read_yaml_file(yaml_path)
    config = Config(**config_data)
    
    image_a, image_b = load_images(config.image_a, config.image_b)

    gray_a, gray_b = bgr_to_gray(image_a, image_b)
    save_images(gray_a, gray_b, './steps/gray_a.jpg', './steps/gray_b.jpg')

    resized_a, resized_b = resize_image(gray_a, gray_b)
    save_images(resized_a, resized_b, './steps/resized_a.jpg', './steps/resized_b.jpg')

    hist_a = calculate_histogram(resized_a)
    hist_b = calculate_histogram(resized_b)
    
    angular_distance = calculate_angular_distance(hist_a, hist_b)
    print(angular_distance)
    
    # Comparar a distância com o threshold
    if angular_distance < config.threshold:
        print("Mesmo produto")
    else:
        print("Produtos diferentes")

    # Concatenar imagens
    concatenated_image = concatenate_images(resized_a, resized_b)

    # Salvar a imagem concatenada
    cv2.imwrite(config.output_location, concatenated_image)
    #print(f"Concatenated image saved at: {config.output_location}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process YAML file.')
    parser.add_argument('yaml_path', type=str, help='Path to the YAML file')
    args = parser.parse_args()
    main(args.yaml_path)