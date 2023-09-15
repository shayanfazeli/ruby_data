from typing import Dict, List, Tuple
import numpy


def polygon_area(polygon_data: List[Dict[str, float]]) -> float:
    x = numpy.array([e['x'] for e in polygon_data])
    y = numpy.array([e['y'] for e in polygon_data])
    return 0.5 * numpy.abs(numpy.dot(x, numpy.roll(y, 1)) - numpy.dot(y, numpy.roll(x, 1)))


def find_central(polygon_data: List[Dict[str, float]]):
    _x_list = [vertex['x'] for vertex in polygon_data]
    _y_list = [vertex['y'] for vertex in polygon_data]
    _len = len(polygon_data)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (int(_x), int(_y))


def find_centroid(polygon_data: List[Dict[str, float]]) -> Tuple[int, int]:
    vertices = polygon_data
    x, y = 0, 0
    n = len(vertices)
    signed_area = 0
    for i in range(len(vertices)):
        x0, y0 = vertices[i]['x'], vertices[i]['y']
        x1, y1 = vertices[(i + 1) % n]['x'], vertices[(i + 1) % n]['y']
        # shoelace formula
        area = (x0 * y1) - (x1 * y0)
        signed_area += area
        x += (x0 + x1) * area
        y += (y0 + y1) * area
    signed_area *= 0.5
    x /= 6 * signed_area
    y /= 6 * signed_area
    return int(x), int(y)


def get_centroid_and_tlbr(polygon_data: List[Dict[str, float]], max_width: int, max_height: int):
    centroid = find_centroid(polygon_data)
    top_left = [centroid[0] - 256, centroid[1] - 256]
    bottom_right = [centroid[0] + 256, centroid[1] + 256]

    if top_left[0] < 0:
        bottom_right[0] -= top_left[0]
        top_left[0] = 0

    if bottom_right[0] > max_width:
        offset = bottom_right[0] - max_width + 1
        bottom_right[0] -= offset
        top_left[0] -= offset

    if top_left[1] < 0:
        bottom_right[1] -= top_left[1]
        top_left[1] = 0

    if bottom_right[1] > max_height:
        offset = bottom_right[1] - max_width + 1
        bottom_right[1] -= offset
        top_left[1] -= offset

    return centroid, (top_left, bottom_right)
