import math

def rotate(point, angle):
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    rpx = a_cos * point[0] + a_sin * point[1]
    rpy = -a_sin * point[0] + a_cos * point[1]
    return rpx, rpy

def cross(a, b, c):
    return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])

def rbbox_to_corners(rbbox):
    # generate clockwise corners and rotate it clockwise
    # 顺时针方向返回角点位置
    cx, cy, x_d, y_d, angle = rbbox
    # a_cos = math.cos(angle)
    # a_sin = math.sin(angle)
    corners_x = [-x_d / 2, -x_d / 2, x_d / 2, x_d / 2]
    corners_y = [-y_d / 2, y_d / 2, y_d / 2, -y_d / 2]
    corners = [0] * 8
    for i in range(4):
        rpx, rpy = rotate(corners_x[i], corners_y[i])
        corners[2 * i] = rpx + cx
        corners[2 * i + 1] = rpy + cy
    return corners


def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0

def point_in_quad(pt_x, pt_y, corners):
    p = [pt_x, pt_y]
    p1 = [corners[0], corners[1]]
    p2 = [corners[2], corners[3]]
    p3 = [corners[4], corners[5]]
    p4 = [corners[6], corners[7]]
    c1 = cross(p1, p2, p)
    c2 = cross(p3, p4, p)
    c3 = cross(p2, p3, p)
    c4 = cross(p4, p1, p)
    return c1 * c2 >= 0 and c3 * c4 >= 0

def line_segment_inter(pts1, pts2, i, j):
    a = [pts1[2 * i], pts1[2 * i + 1]]
    b = [pts1[2 * ((i + 1) % 4)], pts1[2 * ((i + 1) % 4) + 1]]
    c = [pts2[2 * j], pts2[2 * j + 1]]
    d = [pts2[2 * ((j + 1) % 4)], pts2[2 * ((j + 1) % 4) + 1]]

    area_abc = cross(a, b, c) / 2.0
    area_abd = cross(a, b, d) / 2.0
    if area_abc * area_abd > 0:
        return False, [0, 0]

    area_cda = cross(c, d, a) / 2.0
    area_cdb = cross(c, d, b) / 2.0
    if area_cda * area_cdb > 0:
        return False, [0, 0]

    t = area_cda / (area_abd - area_abc)
    dx= t * (b[0] - a[0])
    dy= t * (b[1] - a[1])
    return [a[0] + dx, a[0] + dy]

def line_segment_intersection(pts1, pts2, i, j):
    # pts1, pts2 为corners
    # i j 分别表示第几个交点，取其和其后一个点构成的线段
    # 返回为 tuple(bool, pts) bool=True pts为交点
    A, B, C, D, ret = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    # 叉乘判断方向
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        # 判断方向
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            ret[0] = Dx / DH
            ret[1] = Dy / DH
            return True, ret
    return False, ret


def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    def _cmp(pt, center):
        vx = pt[0] - center[0]
        vy = pt[1] - center[1]
        d = math.sqrt(vx * vx + vy * vy)
        vx /= d
        vy /= d
        if vy < 0:  # 0 to 180
            vx = -2 - vx  # (-1, 1) -> (-3, -1)
        return vx

    if num_of_inter <= 0:
        return
    
    # get center of polygon
    center = [0, 0]
    for i in range(num_of_inter):
        center[0] += int_pts[i][0]
        center[1] += int_pts[i][1]
    center[0] /= num_of_inter
    center[1] /= num_of_inter

    int_pts.sort(key=lambda x: _cmp(x, center))


def area(int_pts, num_of_inter):
    def _trangle_area(a, b, c):
        return abs(corss(a, b, c)) / 2.0

    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += _trangle_area(int_pts[0], int_pts[i + 1], int_pts[i + 2])
    
    return area_val


if __name__ == '__main__':
    rbbox1 = [0, 0, 2, 4, 0]
    rbbox2 = [0, 0, 4, 2, 0]
    corners1 = rbbox_to_corners(rbbox1)
    corners2 = rbbox_to_corners(rbbox2)
    pts, num_pts = [], 0
    for i in range(4):
        point = [corners1[2 * i], corners1[2 * i + 1]]
        if point_in_quadrilateral(point[0], point[1], corners2):
            num_pts += 1
            pts.append(point)
    for i in range(4):
        point = [corners2[2 * i], corners2[2 * i + 1]]
        if point_in_quadrilateral(point[0], point[1], corners1):
            num_pts += 1
            pts.append(point)
    for i in range(4):
        for j in range(4):
            ret, point = line_segment_intersection(corners1, corners2, i, j)
            if ret:
                num_pts += 1
                pts.append(point)
    sort_vertex_in_convex_polygon(pts, num_pts)
    polygon_area = area(pts, num_pts)
    print('area: {}'.format(polygon_area))
