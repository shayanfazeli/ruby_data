import os
import numpy
import json
import tifffile
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from ruby_data.utilities.utils import first_digit_index


def get_whole_slide_filepaths(root: str) -> Dict[str, List[Tuple[str, str]]]:
    whole_slides = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        for e in filenames:
            if e.endswith('.tif'):
                annot = os.path.join(dirpath, os.path.splitext(e)[0] + '.json')
                assert os.path.isfile(annot), annot
                whole_slides[e[:first_digit_index(e)-1]].append(
                    (os.path.join(dirpath, e), annot)
                )
            elif not e.endswith('.json'):
                raise Exception(f"Unknown file extention detected: {os.path.splitext(e)[1]}")
    return whole_slides


def load_whole_slides_in_memory(whole_slides: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Tuple[numpy.ndarray, Any]]]:
    fetched_whole_slides = defaultdict(list)
    for category in whole_slides:
        for data_filepath, meta_filepath in whole_slides[category]:
            with tifffile.TiffFile(data_filepath) as tif:
                data = tif.asarray()
            with open(meta_filepath, 'r') as handle:
                meta = json.load(handle)
        fetched_whole_slides[category].append((data, meta))

    return fetched_whole_slides


def build_item_identifiers(fetched_whole_slides):
    identifiers = []

    for category, category_items in fetched_whole_slides.items():
        for i, (image, annot) in enumerate(category_items):
            for j in range(len(annot['annotations'])):
                identifiers.append((category, i,  # image index
                                    j  # polygon index
                                    ))
    return identifiers

