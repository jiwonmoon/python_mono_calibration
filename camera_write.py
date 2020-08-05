import cv2

image_out_path = "C://Users//mjw31//Desktop//data_DP//stereo_test//output//mono_calibration_image//"
front_image_extension = "camera_"
back_image_extension = ".png"
image_seq_num = 0

cap = cv2.VideoCapture("C://Users//mjw31//Desktop//data_DP//PNC_data//sequences//Camera6//record_2020-08-02-14.25.45.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 220)

while(True):
    ret, image = cap.read()

    if(ret):

        cv2.imshow('image_L', image)
        image_out_name = image_out_path + front_image_extension + str(image_seq_num) + back_image_extension

        key_value = cv2.waitKey(30)
        if key_value == ord('c'):
            print(image_out_name)
            cv2.imwrite(image_out_name, image)
            image_seq_num = image_seq_num + 1

        if key_value == ord('q'):
            break

cv2.destroyAllWindows()