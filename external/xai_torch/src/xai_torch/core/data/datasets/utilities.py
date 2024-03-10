"""
Defines dataset related utility functions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from typing import TYPE_CHECKING, Tuple, Union


def normalize_bbox(bbox: Tuple[int, int, int, int], size: Tuple[int, int]):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def read_ocr_data(ocr_file_path: Union[str, Path]):
    from pathlib import Path

    import bs4

    words = []
    word_bboxes = []
    word_angles = []
    try:
        ocr_file = Path(ocr_file_path)
        if ocr_file.exists() and ocr_file.stat().st_size > 0:
            with open(ocr_file, "r", encoding="utf-8") as f:
                xml_input = eval(f.read())
            soup = bs4.BeautifulSoup(xml_input, "lxml")
            ocr_page = soup.findAll("div", {"class": "ocr_page"})
            image_size_str = ocr_page[0]["title"].split("; bbox")[1]
            w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())
            ocr_words = soup.findAll("span", {"class": "ocrx_word"})
            for word in ocr_words:
                title = word["title"]
                conf = int(title[title.find(";") + 10 :])
                if word.text.strip() == "" or conf < 50:
                    continue

                # get text angle from line title
                textangle = 0
                parent_title = word.parent["title"]
                if "textangle" in parent_title:
                    textangle = int(parent_title.split("textangle")[1][1:3])

                x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
                words.append(word.text.strip())
                word_bboxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
                word_angles.append(textangle)
        else:
            logger.warning(f"Cannot read file: {ocr_file}.")
    except Exception as e:
        logger.exception(f"Exception raised while reading ocr data from file {ocr_file}: {e}")
    return words, word_bboxes, word_angles
