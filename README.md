# Детектирование лиц на фотографиях с помощью сверточных нейронных сетей
## Участники команды
* Кетков Сергей - @sketkov1994
* Кондратьев Никита <nskondratyev@yandex.ru> - @nskondratev
## Описание
В проекте мы рассматриваем задачу детектирования лиц на основе сверточных
нейронных сетей. Для тестирования мы используем две обученные сверточные сети:
* [MTCNN](https://medium.com/wassa/modern-face-detection-based-on-deep-learning-using-python-and-mxnet-5e6377f22674)
* [Tiny Face Detector](https://www.cs.cmu.edu/~peiyunh/tiny/)

Для тестирования используется часть набора данных [Wider face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/): 976 изображений из валидационной выборки.

## Структура проекта
* **DockerFiles** - подмодуль с Docker-образами с предустановленными пакетами для MTCNN
* **mtcnn** - подмодуль с реализацией MTCNN
* **tiny** - подмодуль с реализацией Tiny Face Detector
* **widerface** - тестовая выборка
* **Report. Face Detection with CNN.pdf** - отчет по проделанной работе
* **Slides. Face Detection with CNN.pdf** - презентация
* **main.py** - главный скрипт для запуска моделей на тестовой выборке
* **tiny_fd.m** - MATLAB скрипт с функцией *tiny_fd* для применения Tiny Face Detector к изображению
* **transform_annotations_file.py** - скрипт для обработки файла с аннотацией к тестовой выборке
* **utils.py** - скрипт с вспомогательными функциями
* **wider_face_val_bbx_gt._transformed.txt** - обработанный файл с аннотацией к тестовой выборке

## Требования
Для запуска проекта необходимо иметь установленным:
* Python 3
* Пакеты Python: numpy, mxnet, matlab, cv2
* MATLAB R2016b

## Установка
Загрузка проекта:
```bash
$ git clone --recursive https://github.com/nskondratev/hse-deeplearning-2017.git
```
## Запуск
Для запуска необходимо выполнить скрипт **main.py**. Доступные опции:
* **folder** - папка с изображениями (по умолчанию: widerface),
* **report_filename** - имя файла отчета,
* **model** - используемая модель (возможные значения: *mtcnn*, *tiny_fd*. По умолчанию: *mtcnn*),
* **gt_filename** - путь к файлу с аннотацией к тестовой выборке (по умолчанию: wider_face_val_bbx_gt._transformed.txt).

Пример:
```bash
$ python3 main.py --folder widerface --model mtcnn
```
## Сравнение MTCNN vs Tiny Face Detector
Мы запускали обе модели на тестовой выборке из 976 изображений, которые находятся в папке *widerface*.  
Результаты представлены в таблице:

|                    | Ошибка   | Время работы |
|--------------------|----------|--------------|
| MTCNN              | 32 %     | 3506 s       |
| Tiny Face Detector | 13 %     | 40435 s      |

