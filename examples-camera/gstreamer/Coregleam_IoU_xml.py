import argparse
import time
import os
import csv
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


import xml.etree.ElementTree as ET
time_to_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
def draw_objects(draw, objs, labels, line_width=5):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='green', width=line_width)
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='green')


def save_bbox_to_csv(results, csv_output_path):
    # CSV 파일로 저장합니다.
    with open(csv_output_path, mode='w', newline='') as file:
        fieldnames = ['idx', 'label', 'id', 'score', 'IOU']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # CSV 파일 헤더를 작성합니다.
        writer.writeheader()

        # 결과 리스트의 각 항목을 CSV 파일에 작성합니다.
        for result in results:
            writer.writerow(result)


######################################## ************************
def calculate_iou(rect1, rect2):
    # 사각형 1의 좌표
    x1_rect1, y1_rect1, x2_rect1, y2_rect1 = rect1
    # 사각형 2의 좌표
    x1_rect2, y1_rect2, x2_rect2, y2_rect2 = rect2

    # 겹치는 영역의 좌표 계산
    x_left = max(x1_rect1, x1_rect2)
    y_top = max(y1_rect1, y1_rect2)
    x_right = min(x2_rect1, x2_rect2)
    y_bottom = min(y2_rect1, y2_rect2)

    # 겹치는 영역의 넓이 계산
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # 두 사각형의 넓이 계산
    rect1_area = (x2_rect1 - x1_rect1) * (y2_rect1 - y1_rect1)
    rect2_area = (x2_rect2 - x1_rect2) * (y2_rect2 - y1_rect2)

    # IOU 계산
    iou = intersection_area / float(rect1_area + rect2_area - intersection_area)

    return iou
########################################

def main():
    default_model_dir = os.environ['HOME']
    default_model_dir = os.path.join(default_model_dir, 'google-coral/examples-camera/all_models')
    default_model = 'Coregleam_hotspot_model_500_v2.tflite'
    default_labels = 'Coregleam_hotspot_panel_labels.txt'
    folder_path = '../tool/IoU/hotspot/image'               #####################################################   1. hotspot , 2. coldspot 
    index = 0


    xml_folder_path = '../tool/IoU/hotspot/xml'           #####################################################   1. hotspot , 2. coldspot

    # 결과를 저장할 빈 리스트를 생성합니다.
    results = []

    for image in os.listdir(folder_path):

        print(image)
        ####################################### ************
        img_num = int(image[8:-4])
        hotspot_path = xml_folder_path + '/' + 'hotspot_'+ '{}'.format(img_num) + '.xml'


        tree = ET.parse(hotspot_path)
        root = tree.getroot()

        # 좌표 숫자만 저장할 리스트 초기화
        coordinates = []

        # 좌표 값 불러오기
        for obj in root.findall('.//object'):
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # 좌표 숫자만 리스트에 저장
            coordinates.extend([xmin, ymin, xmax, ymax])

        ######################################


        output = '../tool/result_image/hotspot_result/hotspot_{}.jpg'.format(img_num)   #####################################################   1. hotspot , 2. coldspot
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--model', default=os.path.join(default_model_dir, default_model))
        parser.add_argument('--input', default=os.path.join(folder_path, image))
        parser.add_argument('--labels', default=os.path.join(default_model_dir, default_labels))
        parser.add_argument('-t', '--threshold', type=float, default=0.4,
                            help='Score threshold for detected objects')
        parser.add_argument('--output', default=os.path.join(default_model_dir, output))
        index += 1
        parser.add_argument('-c', '--count', type=int, default=5,
                            help='Number of times to run inference')
        args = parser.parse_args()

        labels = read_label_file(args.labels) if args.labels else {}
        interpreter = make_interpreter(args.model)
        interpreter.allocate_tensors()

        image = Image.open(args.input)
        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

        print('----INFERENCE TIME----')
        print('Note: The first inference is slow because it includes',
            'loading the model into Edge TPU memory.')
        for _ in range(args.count):
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_objects(interpreter, args.threshold, scale)
            print('%.2f ms' % (inference_time * 1000))

        print('-------RESULTS--------')
# #####################################  ***********************************
        idx_4 = 0
        while len(coordinates) > 4*idx_4:
            xmin, ymin, xmax, ymax = coordinates[0+4*idx_4], coordinates[1+4*idx_4], coordinates[2+4*idx_4], coordinates[3+4*idx_4]
            idx_4 += 1
            ImageDraw.Draw(image).rectangle([(xmin, ymin), (xmax, ymax)], outline='blue', width=5)

#####################################  

        
        if not objs:
            print('No objects detected')
        else:
            for obj in objs:
                # print(obj)
                # print(labels.get(obj.id, obj.id))
                # print('  id:    ', obj.id)
                # print('  score: ', obj.score)
                # print('  bbox:  ', obj.bbox)

###############################################  **************************************
                idx_4 = 0
                while len(coordinates) > 4*idx_4:
                    xmin, ymin, xmax, ymax = coordinates[0+4*idx_4], coordinates[1+4*idx_4], coordinates[2+4*idx_4], coordinates[3+4*idx_4]
                    rect1 = (xmin, ymin, xmax, ymax)
                    rect2 = (obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax)
                    idx_4 += 1

                    iou_value = calculate_iou(rect1, rect2)
#####################################
                # 객체 정보를 results 리스트에 추가합니다.
                    result_dict = {
                        'idx' : img_num,
                        'label': labels.get(obj.id, obj.id),
                        'id': obj.id,
                        # 'score': obj.score,
                        # 'bbox_xmin': obj.bbox.xmin,
                        # 'bbox_ymin': obj.bbox.ymin,
                        # 'bbox_xmax': obj.bbox.xmax,
                        # 'bbox_ymax': obj.bbox.ymax,
                        # 'gt_xmin' : xmin,
                        # 'gt_ymin' : ymin,
                        # 'gt_xmax' : xmax,
                        # 'gt_ymax' : ymax,
                        'IOU' : iou_value,

                    }
                    results.append(result_dict)

        if args.output:
            image = image.convert('RGB')
            draw_objects(ImageDraw.Draw(image), objs, labels)
            image.save(args.output)
            image.show()


    # CSV 파일로 bbox 정보 저장
    csv_output_path = f'/home/mendel/google-coral/examples-camera/tool/result_csv/IoU/bbox_coordinates_{time_to_time}_xml.csv'

    save_bbox_to_csv(results, csv_output_path)



if __name__ == '__main__':
    main()
