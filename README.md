# python_mono_calibration

input: 56(front)

- K, d 계산
1. mono camea capture 				<camera_write.py>
2. calbration					<camera_calibration.py>
3. save K, d					<camera_calibratin.py>

- stereo [R|t] 계산 (using K,d)
1. stereo camera capture				<stereo_write.py>
2. calibration					<stereo_calibraion_PNC.py>
3. save stereo[R|t]					<stereo_calibraion_PNC.py>

- stereo distance estimation (using K,d, stereo[R|t])
1. apply algorithms					<stereo_PNC_test.py>
