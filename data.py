def get_calibration_paths():
    res = []
    for i in range(20):
        res.append('./camera_cal/calibration{}.jpg'.format(i + 1))
    return res


def get_test_paths():
    res = []
    for i in range(6):
        res.append('./test_images/test{}.jpg'.format(i + 1))
    return res


def get_straight_line_paths():
    res = []
    for i in range(2):
        res.append('./test_images/straight_line{}.jpg'.format(i + 1))
    return res
