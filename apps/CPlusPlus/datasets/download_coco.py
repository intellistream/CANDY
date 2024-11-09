import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Python.Embedding.datasetEmbedding import cocoEmbedding

def getCoco(cocoPath):
    # chick if coco dataset exist, if not, download
    if os.path.exists(cocoPath + "/" +"coco1.txt" ):
        print('download of coco is done')
    else:
        # os.system("cd "+cocoPath+ '&& wget http://images.cocodataset.org/zips/train2017.zip')
        # os.system("cd "+cocoPath+'&& wget http://images.cocodataset.org/zips/val2017.zip')
        # print("download coco2017....")
        os.system("cd "+cocoPath+'&& touch coco1.txt')

    # chick if unziped coco dataset exist, if not, upzip
    if os.path.exists(cocoPath + "/" +"coco2.txt" ):
        print('unzip of coco is done')
    else:
        os.system("cd "+cocoPath+'&& unzip train2017.zip && unzip val2017.zip')
        os.system("cd "+cocoPath+'&& touch coco2.txt')


def main():
    # download dependencies

    # check if all required files exist
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    targetPath=exeSpace+"datasets/coco"
    # checkList = ('datasets/coco/query_image.fvecs',
    #     'datasets/coco/query_captions.fvecs',
    #     'datasets/coco/query_shuffle.fvecs',
    #     'datasets/coco/query_shuffle.fvecs',
    #     'datasets/coco/data_image.fvecs',
    #     'datasets/coco/data_captions.fvecs',
    #     'datasets/coco/data_shuffle.fvecs',
    #     'datasets/coco/data_shuffle.fvecs')
    # files = 0
    # for i in checkList:
    #     if(os.path.exists(exeSpace+i)):
    #         files = files +1
    if(os.path.exists(targetPath)) :
        print('#####   skip generation of coco!   #####')
    else:
        targetPathBase = exeSpace + 'datasets/coco'
        os.makedirs(targetPathBase, exist_ok=True)
        os.system('rm -rf ' + targetPathBase + "/*.fvecs")
        getCoco(targetPathBase)



if __name__ == "__main__":
    main()

