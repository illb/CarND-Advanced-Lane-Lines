def get_calibration_paths():
    res = []
    for i in range(20):
        res.append('./camera_cal/calibration{}.jpg'.format(i + 1))
    return res


def get_test_paths():
    res = []
    for i in range(6):
        res.append('./test_images/test{}.jpg'.format(i + 1))

    for i in range(7):
        res.append('./test_images/hard_test{}.jpg'.format(i + 1))

    for i in range(2):
        res.append('./test_images/straight_lines{}.jpg'.format(i + 1))

    return res

def get_video_paths():
    res = ["project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]
    return res

