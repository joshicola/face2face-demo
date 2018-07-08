import os
import glob
import cv2
import dlib
import time
import argparse
import numpy as np
from imutils import video

import models
import NonLinearLeastSquares
from drawing import drawProjectedShape
import utils

DOWNSAMPLE_RATIO = 4

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
lockedTranslation = False

def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    # detekcja twarzy
    dets = detector(scaledImg, 1)

    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = dlib.rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale),
                                  int(det.bottom() / imgScale))

        # detekcja punktow charakterystycznych twarzy
        dlibShape = predictor(img, faceRectangle)

        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        # transpozycja, zeby ksztalt byl 2 x n a nie n x 2, pozniej ulatwia to obliczenia
        shape2D = shape2D.T

        shapes2D.append(shape2D)

    return shapes2D


def main():
    os.makedirs('original', exist_ok=True)
    os.makedirs('landmarks', exist_ok=True)
    mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("candide.npz")
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])
    files=[]
    for fl_type in ['png','jpg']:
        [files.append(fl) for fl in glob.glob(args.Directory+'/*.'+fl_type)]
        
    print(len(files))

    count = 0
    imgParams={}
    for fl in files:
        frame=cv2.imread(fl)
        print(fl)
        frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        black_image = np.zeros(frame.shape, np.uint8)

        t = time.time()

        # Perform if there is a face detecte
        shapes2D = getFaceKeypoints(frame, detector, predictor)
        if shapes2D is None:
            continue
        if len(shapes2D) == 1:
            # 3D model parameter initialization
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shapes2D[0][:, idxs2D])

            # 3D model parameter optimization
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual,
                                                    projectionModel.jacobian, (
                                                        [mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]],
                                                        shapes2D[0][:, idxs2D]), verbose=0)

            drawProjectedShape(black_image, [mean3DShape, blendshapes], projectionModel, mesh, modelParams,
                       lockedTranslation)

            # Display the resulting frame
            imgParams[fl]=modelParams
            count += 1
            print(count)
            cv2.imwrite("original/{}.png".format(count), frame)
            cv2.imwrite("landmarks/{}.png".format(count), black_image)

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        else:
            print("No face detected")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dir', dest='Directory', type=str, help='Name of the directory of files.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
