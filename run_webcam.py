import argparse
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import video

import models
import NonLinearLeastSquares
from drawing import drawProjectedShape
import utils
from generate_train_data import getFaceKeypoints as getFaceKeypoints


CROP_SIZE = 256
DOWNSAMPLE_RATIO = 4
lockedTranslation = False


def reshape_for_polyline(array):
    """Reshape image so that it works with polyline."""
    return np.array(array, np.int32).reshape((-1, 1, 2))


def resize(image):
    """Crop and resize image for pix2pix."""
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def main():
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = None

    if writer is None:
        print("Starting video writer")
        writer = cv2.VideoWriter("./out.mp4", fourcc, 30.0, (512, 256))

        if writer.isOpened():
            print("Writer succesfully opened")
        else:
            writer = None
            print("Writer opening failed")
    else:
        print("Stopping video writer")
        writer.release()
        writer = None

    # TensorFlow
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # OpenCV
    # cap = cv2.VideoCapture(args.video_source)
    cap = cv2.VideoCapture(args.video_dir)
    fps = video.FPS().start()

    mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("candide.npz")
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        # resize image and detect face
        frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)

        #get frame face label
        black_image = np.zeros(frame.shape, np.uint8)

        shapes2D = getFaceKeypoints(frame, detector, predictor)
        if shapes2D is None:
            continue
        # 3D model parameter initialization
        modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shapes2D[0][:, idxs2D])

        # 3D model parameter optimization
        modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual,
                                                        projectionModel.jacobian, (
                                                            [mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]],
                                                            shapes2D[0][:, idxs2D]), verbose=0)

        drawProjectedShape(black_image, [mean3DShape, blendshapes], projectionModel, mesh, modelParams,
                           lockedTranslation)

        # generate prediction
        combined_image = np.concatenate([resize(black_image), resize(frame_resize)], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        image_normal = np.concatenate([resize(frame_resize), image_bgr], axis=1)
        image_landmark = np.concatenate([resize(black_image), image_bgr], axis=1)

        if args.display_landmark == 0:
            cv2.imshow('frame', image_normal)
        else:
            cv2.imshow('frame', image_landmark)

        if writer is not None:
            writer.write(image_normal)

        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    sess.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-src', '--source', dest='video_source', type=int,
    #                     default=0, help='Device index of the camera.')
    parser.add_argument('--show', dest='display_landmark', type=int, default=0, choices=[0, 1],
                        help='0 shows the normal input and 1 the facial landmark.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    parser.add_argument('--tf-model', dest='frozen_model_file', type=str, help='Frozen TensorFlow model file.')
    parser.add_argument('--video-dir', dest='video_dir', type=str, help='video file path')

    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
