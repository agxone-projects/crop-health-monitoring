import json
import funcy
from sklearn.model_selection import train_test_split

import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

from model import CornNetv2


def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def split_coco_annotation(annotations, split_ratio, train_json='train_anno.json', test_json='test_anno.json', is_having=True):
    with open(annotations, 'rt', encoding='UTF-8') as anno:
        coco = json.load(anno)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = funcy.lmap(
            lambda a: int(a['image_id']), annotations)

        if is_having:
            images = funcy.lremove(
                lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=split_ratio)

        save_coco(train_json, x, filter_annotations(
            annotations, x), categories)
        save_coco(test_json, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(
            len(x), train_json, len(y), test_json))


if __name__ == '__main__':

    # spliting the annotations
    split_coco_annotation('../../data/interim/annotations.json', 0.9,
                          '../../data/interim/train_annotations.json',
                          '../../data/interim/test_annotations.json')

    # registering the train and test datasets
    register_coco_instances("corn_crops_train", {},
                            '../../data/interim/train_annotations.json',
                            '../../data/interim')
    register_coco_instances("corn_crops_test", {},
                            '../../data/interim/test_annotations.json',
                            '../../data/interim')

    # retrieving metadata and dicts
    corn_crop_metadata = MetadataCatalog.get("corn_crops_train")
    train_dataset_dicts = DatasetCatalog.get("corn_crops_train")
    test_dataset_dicts = DatasetCatalog.get("corn_crops_test")

    model = CornNetv2(
        weights="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
        n_classes=1,
        train_sets=("corn_crops_train", ),)
    model.compile(n_iter=1000, resume=True)

    print('Training the Model\n')
    output_dir = '../../models/cornnetv2'
    model.fit(output_dir)

    print('Evaluating using the Model')
    model.evaluate("corn_crops_test")
