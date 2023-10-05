import numpy as np

def isPosInRect(point, box):
    px, py = point
    height, width, c_x, c_y, _ = box
    hh = height / 2
    hw = width / 2
    if px > c_x - hw and px < c_x + hw and py > c_y - hh and py < c_y + hh:
        return True
    return False

def isPosInRotationRect(point, box):
    px, py = point
    height, width, c_x, c_y, yaw = box
    hw = width / 2
    hh = height / 2
    yaw = yaw * (np.pi / 180)
    px2 = c_x + (px - c_x) * np.cos(yaw) - (py - c_y) * np.sin(yaw)
    px2 = c_y + (px - c_x) * np.sin(yaw) + (py - c_y) * np.cos(yaw)
    if px2 > c_x - hw and px2 < c_x + hw and px2 > c_y - hh and px2 < c_y + hh:
        return True
    return False

if __name__ == "__main__":
    point = [0.5, 1.0]
    box = [1, 1, 10, 10.8, 0.5]
    print(isPosInRect(point, box))
    print(isPosInRotationRect(point, box))
