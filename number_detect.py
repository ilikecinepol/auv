import cv2
import numpy as np


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return img_bin


def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def count_corners(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return len(approx)


def analyze_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image from path '{img_path}'")
        return

    img_bin = preprocess_image(img)
    contours = find_contours(img_bin)

    found_number = None  # Инициализируем переменную для определения числа на изображении

    for contour in contours:
        corners = count_corners(contour)
        # Отображаем контуры с числом углов
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Отображение числа углов рядом с контуром
            cv2.putText(img, str(corners), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if corners == 4:
            found_number = 1
        elif corners == 12:
            # Находим прямоугольник и обрезаем изображение
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x, y, w, h = cv2.boundingRect(box)
            cropped_img = img[y:y + h, x:x + w]
            # Проверяем среднее значение пикселей в 67 строке обрезанного изображения
            cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            row_67 = cropped_gray[67, :13]  # Берем первые 13 элементов 67 строки
            mean_value = np.mean(row_67)
            if mean_value == 255:
                found_number = 3
            else:
                found_number = 2

    # Выводим результат
    if found_number is not None:
        print(f"Found number: {found_number}")
    else:
        print("Number not identified.")

    # Отображаем и сохраняем изображение с контурами и числами углов
    cv2.imshow('Contours with Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Пример вызова функции анализа изображения
if __name__ == "__main__":
    img_path = 'C:/Users/drmma/Downloads/murArtem/Screenshot_20.png'
    analyze_image(img_path)
