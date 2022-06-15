import imghdr
import random
import shutil
from seetaface.api import *

def get_all_images(input_dir):
    """
    遍历获取输入文件夹内的所有图像文件。

    :param input_dir: str, input directory
    :return image_list: list, image list
    """
    image_list = []
    results = os.walk(input_dir)
    for result in results:
        dir_path, _, image_files = result
        for image_file in image_files:
            if imghdr.what(os.path.join(dir_path, image_file)):  # this is an image file
                image_list.append(os.path.join(dir_path, image_file))
    return image_list


def random_rename(image_name):
    """
    通过在图片名后追加随机数字的方式更改图片文件名称。

    :param image_name: origin image name, str
    :return image_name: new image name, str
    """
    index = random.randint(0, 9999)
    prefix, suffix = os.path.splitext(image_name)
    image_name = prefix + "_%4d" % index + suffix
    return image_name


if __name__ == "__main__":
    init_mask = FACE_DETECT | FACERECOGNITION | LANDMARKER5
    seetaFace = SeetaFace(init_mask)
    # 这样从计算复杂度上，能够比最小人脸设置为20时，提速到16倍
    seetaFace.SetProperty(DetectProperty.PROPERTY_MIN_FACE_SIZE, 80)
    # 设置最小检测人脸阈值0.9，大于0.9得分的人脸才返回
    seetaFace.SetProperty(DetectProperty.PROPERTY_THRESHOLD, 0.9)

    # 一张仅包含要查看人脸的相片的路径
    yourFaceImageFileLocation = "D:\\tempPic\\IMG_8311.jpeg"
    yourFaceImage = cv2.imread(yourFaceImageFileLocation)
    detect_result1 = seetaFace.Detect(yourFaceImage)
    if detect_result1.size <= 0:
        print("the image have no people face")
    face = detect_result1.data[0].pos
    points = seetaFace.mark5(yourFaceImage, face)
    yourFaceFeature = seetaFace.Extract(yourFaceImage, points)

    # 要分类相片的文件夹
    inputDir = "D:\\tempGraduation"
    imageList = get_all_images(inputDir)
    i = 0
    for imageFile in imageList:
        i = i + 1
        image = cv2.imread(imageFile)
        detectResult1 = seetaFace.Detect(image)
        if detectResult1.size <= 0:
            continue
        for index in range(detectResult1.size):
            face = detectResult1.data[index].pos
            points = seetaFace.mark5(image, face)
            feature = seetaFace.Extract(image, points)
            similar1 = seetaFace.CalculateSimilarity(yourFaceFeature, feature)
            if similar1 > 0.7:
                if not os.path.exists(os.path.join("IMAGE")):
                    os.makedirs(os.path.join("IMAGE"))
                _, image_name = os.path.split(imageFile)
                if os.path.exists(os.path.join("IMAGE", image_name)):
                    image_name_person = random_rename(image_name)
                else:
                    image_name_person = image_name
                try:
                    shutil.copy(imageFile, os.path.join("IMAGE", image_name_person))
                except Exception:
                    print("Error happened when building image link of %s" % imageFile)
                break
        print(str(i) + "/" + str(len(imageList)))
