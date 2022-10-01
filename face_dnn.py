
import time
import cv2
import numpy as np
import time
import traceback
import targeting_tools as tt

USE_CUDA = True

iterations = 1
start = time.perf_counter()

print("running")


print("new keyboard test")



def show_detection(image, face, confidence):
    # draw rectangles
    # print(faces)
    (x1, y1, x2, y2) = face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    # text shadow
    cv2.putText(image, '{:0.2%}'.format(confidence), (x1+1, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
    # text
    cv2.putText(image, '{:0.2%}'.format(confidence), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 2, cv2.LINE_AA,
                bottomLeftOrigin=False)
    return image


#cap = cv2.VideoCapture('./video/gentleman.mp4')  # video file
cap = cv2.VideoCapture(0)  # usb camera
confidence_threshold = 0.69


# output image dimensions
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# load DNN pre-trained model
# net = cv2.dnn.readNetFromCaffe('res10_300x300_ssd_deploy.prototxt', caffeModel='res10_300x300_ssd_iter_140000_fp16.caffemodel')
net = cv2.dnn.readNetFromCaffe('./ssd/res10_300x300_ssd_deploy.prototxt',
                               caffeModel='./ssd/res10_300x300_ssd_iter_140000.caffemodel')
# use cuda
if USE_CUDA:
    _ = net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    _ = net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while cv2.waitKey(1) != ord(' ') and cap.isOpened():  # space bar
    # iterations = 1
    # start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    # massage our input data to fit model
    blob = cv2.dnn.blobFromImage(frame, 1.0, size=(600,500), mean=[120.0, 130.0, 140.0], swapRB=False, crop=False)

    # feed our blob as input to our net model
    net.setInput(blob)

    # inference, forward
    detections = net.forward()

    # iterate over all found faces in detections
    detected_faces_count = 0
    for i in range(detections.shape[2]):
        # get confidence value
        confidence = detections[0,0,i,2]

        # only consider this item if confidence > acceptable criteria
        if confidence > confidence_threshold:
            # count number of faces detected
            detected_faces_count = detected_faces_count + 1
            # get box coordinates for this face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # draw rectangle and confidence value
            show_detection(frame, box.astype("int"), confidence)

    # end = time.perf_counter()
    # print("Face Detection DNN: {0} msec".format(((end - start) / iterations) * 1000))
    def run():

        try:

            left_camera_source = 2
            pixel_width = 640
            pixel_height = 480
            angle_width = 78
            angle_height = 64  # 63
            frame_rate = 20

            ct1 = tt.Camera_Thread()
            ct1.camera_source = left_camera_source
            ct1.camera_width = pixel_width
            ct1.camera_height = pixel_height
            ct1.camera_frame_rate = frame_rate

            ct1.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            ct1.start()
            angler = tt.Frame_Angles(pixel_width, pixel_height, angle_width, angle_height)
            angler.build_frame()

            # motion detection
            targeter1 = tt.Frame_Motion()
            targeter1.gaussian_blur = 15  # blur (must be positive and odd)
            targeter1.threshold = 15
            targeter1.dilation_value = 6
            targeter1.dilation_kernel = np.ones((targeter1.dilation_value, targeter1.dilation_value), np.uint8)
            targeter1.erosion_iterations = 0
            targeter1.dilation_iterations = 4
            targeter1.contour_min_area = 1  # percent of frame area
            targeter1.contour_max_area = 80  # percent of frame area
            targeter1.targets_max = 5
            targeter1.target_on_contour = True  # False = use box size
            targeter1.target_return_box = False  # (x,y,bx,by,bw,bh)
            targeter1.target_return_size = True  # (x,y,%frame)
            targeter1.contour_draw = True
            targeter1.contour_box_draw = True
            targeter1.targets_draw = 1

            time.sleep(0.5)

            klen = 1

            x1k, y1k, = [], []

            x1m, y1m = 0, 0

            X, Y, = 0, 0

            while 1:
                frame1 = ct1.next(black=True, wait=1)

                targets1 = targeter1.targets(frame1)

                if not targets1:

                    x1k, y1k, = [], []

                else:
                    x1, y1, s1 = targets1[0]
                    x1k.append(x1)
                    y1k.append(y1)

                    if len(x1k) >= klen:
                        x1k = x1k[-klen:]
                        y1k = y1k[-klen:]

                        x1m = sum(x1k) / klen
                        y1m = sum(y1k) / klen

                        X, Y = xlangle, ylangle = angler.angles_from_center(x1m, y1m, top_left=True, degrees=True)

                angler.frame_add_crosshairs(frame1)

                fps1 = int(ct1.current_frame_rate)
                text = 'X: {:3.1f}*\nY: {:3.1f}*\nFPS: {}'.format(X, Y, fps1)
                lineloc = 0
                lineheight = 30
                for t in text.split('\n'):
                    lineloc += lineheight
                    cv2.putText(frame1,
                                t,
                                (10, lineloc),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                                False)

                if 1: targeter1.frame_add_crosshairs(frame1, x1m, y1m, 48)






                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty('face dnn', cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('q'):
                    break
                elif key != 255:
                    print('key press:' [chr(key)])
        except:
            print(traceback.format_exc())


    cv2.imshow('Face DNN', frame)
    print("running...")

end = time.perf_counter()
print("Face Detection DNN: {0} msec".format(((end-start) / iterations) * 1000))
cv2.destroyAllWindows()