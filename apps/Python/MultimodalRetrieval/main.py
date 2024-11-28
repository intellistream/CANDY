import sys
import numpy as np
import torch

from Tools.RWTool import Read_ivecs, Read_fvecs, Write_ivecs, Calculate_recall
from Tools.Client import DBClient

from pycandy import ConfigMap
from pycandy import DataLoaderTable


def main():

    # 1.1 read config file.
    defaultFile = "config.csv"
    fileName = sys.argv[1] if len(sys.argv) >= 2 else defaultFile
    inMap = ConfigMap()

    if inMap.from_file(fileName):
        print(f"[Python INFO] 1.1  Config loaded from file: \033[1;32m{fileName}\033[0m")
    else:
        print("-" * 72)
        print(str("1.1 Failed to load config from file: \"" + fileName + "\"").center(72));
        print("-" * 72)
        return -1

    # 1.2 load dataLoader and load data
    dataLoaderTable = DataLoaderTable()
    dataLoaderTag = inMap.try_string("dataLoaderTag", "random", True)
    dataLoader = dataLoaderTable.find_data_loader(dataLoaderTag)

    if dataLoader is None:
        print("-" * 72)
        print(str("1.2 Data loader not found:  \"" + dataLoaderTag + "\"").center(72))
        print("-" * 72)
        return -1
    else:
        print(f"[Python INFO] 1.2  Data loader: \033[1;32m{str(dataLoaderTag).upper()}dataloader\033[0m is ready")

    # set dataLoader`s param
    dataLoader.setConfig(inMap)
    vecDim = inMap.try_i64("vecDim", 768, True)

    # 1.3, 1.4 get Index data and Query data
    dataTensorAll = dataLoader.getData().nan_to_num(0)
    print("[Python INFO] 1.3  Index Data loaded: Dimension = " + str(dataTensorAll.size(1)) + ", DataNum = " + str(dataTensorAll.size(0)))

    queryTensor = dataLoader.getQuery().nan_to_num(0)
    print("[Python INFO] 1.4  Query Data loaded: Dimension = " + str(queryTensor.size(1)) + ", DataNum = " + str(queryTensor.size(0)))

    print("=" * 50)

    # 2.1 Initialize vector database
    indexTag = str(inMap.try_string("indexTag", "knn", True)).lower() + "search"
    client = DBClient(vecDim, indexTag, inMap)
    print(f"[Python INFO] 2.1  The vector database has been successfully initialized, we will use \033[1;32m{indexTag}\033[0m as our algorithm")

    # 2.2 load initial vector
    initialRows = inMap.try_i64("initialRows", 0, True)
    print(f"[Python INFO] 2.2  Load initial vector, Num = {initialRows}")

    initialTag = 0
    if(initialRows > 0):
        initialTag = 1
        dataTensorInitial = dataTensorAll[:initialRows]
        dataTensorStream = dataTensorAll[initialRows:]
        if client.load_batch_tensor(dataTensorInitial):
            print(f"[Python INFO] 2.2  Loading completed")
        else:
            print("-" * 72)
            print(str("Loading failed").center(72))
            print("-" * 72)
            return -1

    print("=" * 50)

    # 3.1 Input the streaming tensor into the database
    if(initialTag == 0):
        dataTensorStream = dataTensorAll

    batchSize = inMap.try_i64("batchSize", dataTensorStream.size(0), True)

    startRow = 0
    endRow = startRow + batchSize
    aRows = dataTensorStream.size(0)


    print("[Python INFO] 3.1  Streaming now !!!")
    processedOld = 0
    batch_count=0
    average_recall=0
    while startRow < aRows:
        sub_batch = dataTensorStream[startRow:endRow]

        # Insert the batch into the index
        if not client.load_batch_tensor(sub_batch):
            print(f"[Python INFO] 2.2  Loading completed")

        # Update the row indices for the next batch
        startRow += batchSize
        endRow += batchSize
        batch_query = queryTensor[startRow:endRow]
        if endRow > aRows:
            endRow = aRows

        # Progress calculation
        if startRow > aRows:
            startRow = aRows
        processed = startRow * 100.0 / aRows
        if processed - processedOld >= 1.0:
            print(f"Done {processed:.2f}% ({startRow}/{aRows})")
            processedOld = processed
    #     if startRow<endRow:
    #         batch_count+=1
    #         Gt = gt[startRow:endRow]
    #         result = client.get_batch_tensors(batch_query,K)
    #         average_recall += Calculate_recall(Gt, result, K)
    # #
    # print(f"Streaming recall@1: {average_recall/batch_count:.4f}")
    # print("[Python INFO] 3.2  Streaming has ended")

    # 3.3 Input the query and get the result
    result = client.get_batch_tensors(queryTensor,K)

    # 3.4 Get the gt
    groundTruthTag = inMap.try_i64("groundTruthTag", 0, True)
    if groundTruthTag == 0:
        print("[Python INFO] 3.4  Ground truth does not exist, so it`s time to create it")
        client_gt = DBClient(vecDim, 'knnsearch', inMap)
        client_gt.load_batch_tensor(dataTensorAll)
        gt_path = "/mnt/f/New/CANDY/apps/Python/MultimodalRetrieval/Datasets/gt_features.fvecs" #根据数据集提供的GT去暴力搜索相似度高的vector
        gt_tensor=torch.tensor((Read_fvecs(gt_path)))
        gt = client_gt.get_batch_tensors(gt_tensor,K)

    else:
        print("[Python INFO] 3.4  Ground truth exists")
        client_gt = DBClient(vecDim, 'knnsearch', inMap)
        client_gt.load_batch_tensor(dataTensorAll)
        file_path = inMap.try_string("groundtruthPath", "/Datasets/gt_features.fvecs",True)
        vectors_gt = torch.tensor(Read_fvecs(file_path))
        gt = client_gt.get_batch_tensors(vectors_gt, K)

    print("=" * 50)


    # 4.1 Calculate metrics "Recall"
    # positiveNum = 0
    # for i in range(len(result)):
    #     if result[i] == gt[i]:
    #         positiveNum += 1
    # recall = (positiveNum / len(result) * 100)
    # (4181, 1)
    # (N, 1, K)
    # result: (N, K) gt[m] = K{indexID*K}
    # gt:(N, K) (gt[m] ∩ result[m])

    print("gt[0]: ",gt[0])
    print("result[0]: ", result[0])
    recall = Calculate_recall(gt, result, K)
    print(f"[Python INFO] 4.1  RECALL = \033[1;32m{recall*100:.4f}%\033[0m")

    return 0

if __name__ == "__main__":
    K=1 #Top K
    main()