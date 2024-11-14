import os
import sys
from datasets import load_dataset
from datasets.config import HF_DATASETS_TRUST_REMOTE_CODE


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Embedding import flavaEmbedding


exeSpace = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) + "/"
targetPathBase = exeSpace + 'datasets/coco'
PATH_TO_IMAGE_FOLDER = targetPathBase


def create_full_path(example):
    """Create full path to image using `base_path` to COCO2017 folder."""
    example["image_path"] = os.path.join(
        PATH_TO_IMAGE_FOLDER, example["file_name"])
    return example

def getcoco():
    # This will get coco2017 meta data from huggingface, next note will help you if you need.
    # export HF_ENDPOINT="https://hf-mirror.com"
    dataset = load_dataset("phiyodr/coco2017")
    dataset = dataset.map(create_full_path)
    return dataset


# embed images and texts
def embedding():
    # get raw dataset
    dataset = getcoco()
    print(PATH_TO_IMAGE_FOLDER)
    exeSpace = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) + "/"
    targetPathBase = exeSpace + 'datasets/coco'
    # generate queries
    captions = dataset['validation']['captions']
    captionFirst = [caption[0] for caption in captions[:len(captions)]]
    print(captionFirst[1])
    caps = flavaEmbedding.encode_captions(captionFirst, 500, 128,
                           targetPathBase + '/query_captions.fvecs')
    images = flavaEmbedding.encode_images_to_fvecs(
        dataset['validation']['image_path'], 500, 16, targetPathBase + '/query_image.fvecs')
    shuffleQuery = flavaEmbedding.shuffle_and_save_fvecs(targetPathBase + '/query_captions.fvecs',
                                          targetPathBase + '/query_image.fvecs',
                                          targetPathBase + '/query_shuffle.fvecs')
    appendQuery = flavaEmbedding.append_fvecs(targetPathBase + '/query_captions.fvecs', targetPathBase + '/query_image.fvecs',
                               targetPathBase + '/query_append.fvecs')

    # generate database
    captions = dataset['train']['captions']
    captionFirst = [caption[0] for caption in captions[:len(captions)]]
    print(captionFirst[1])
    caps = flavaEmbedding.encode_captions(captionFirst, 100000, 128,
                           targetPathBase + '/data_captions.fvecs')
    images = flavaEmbedding.encode_images_to_fvecs(
        dataset['train']['image_path'], 100000, 16, targetPathBase + '/data_image.fvecs')
    shuffleData = flavaEmbedding.shuffle_and_save_fvecs(targetPathBase + '/data_captions.fvecs', targetPathBase + '/data_image.fvecs',
                                         targetPathBase + '/data_shuffle.fvecs')
    appendData = flavaEmbedding.append_fvecs(targetPathBase + '/data_captions.fvecs', targetPathBase + '/data_image.fvecs',
                              targetPathBase + '/data_append.fvecs')


if __name__ == "__main__":
    embedding()
