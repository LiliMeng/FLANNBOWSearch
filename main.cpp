#include <iostream>
#include <ctime>
#include <Vocabulary.h>

using namespace std;

using namespace cv;

int main()
{
    int imgNum=300;

    vector<Mat>  imgVec;
    imgVec.resize(imgNum);

    vector<string> nameVec;
    nameVec.resize(imgNum);

    vector<vector<KeyPoint> > keyPointsVec;
    keyPointsVec.resize(imgNum);

    vector<Mat> descriptorsVec;
    descriptorsVec.resize(imgNum);

    for(int i=0; i<imgNum; i++)
    {
        char fileName[1024] ={NULL};

        sprintf(fileName, "/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_%06d.ppm", i);

        nameVec[i]=string(fileName);

        imgVec[i]=imread(nameVec[i], CV_LOAD_IMAGE_GRAYSCALE);
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector(minHessian);

    SurfDescriptorExtractor extractor;

    vector<unsigned int> labels;
    for(int i=0; i<imgNum; i++)
    {
        detector.detect(imgVec[i], keyPointsVec[i]);

        extractor.compute(imgVec[i], keyPointsVec[i], descriptorsVec[i]);
        for(int j = 0; j<descriptorsVec[i].rows; j++)
        {
            labels.push_back(i);
        }
    }

    Mat all_descriptors;

    for(int i = 0; i<descriptorsVec.size(); i++)
    {
        all_descriptors.push_back(descriptorsVec[i]);
    }

    assert(labels.size() == all_descriptors.rows);
    cout<<"all_descriptors.rows "<<all_descriptors.rows<<endl;
    cout<<"hahha1 "<<endl;
    Vocabulary vocab;

    vocab.indexedDescriptors_ = all_descriptors;


    vector<KeyPoint> newKeypoints;
    Mat newDescriptors;

    ///add new image to the randomized kd tree
    {
        string newImageName="/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000350.ppm";
        Mat newImg=imread(newImageName, CV_LOAD_IMAGE_GRAYSCALE);
        detector.detect(newImg, newKeypoints);
        extractor.compute(newImg, newKeypoints, newDescriptors);
        cout<<"newDescriptors.rows: "<<newDescriptors.rows<<endl;
    }

    vocab.notIndexedDescriptors_ = newDescriptors;

    ///clustering
    int clustersNum;
    Mat clusters(15000,64,CV_32F);
    //Mat float_all_descriptors;


    clustersNum=vocab.clustering(all_descriptors, clusters);
    cout<<"clustersNum  "<<clustersNum<<endl;

    ///flann build tree
    clock_t begin1 = clock();
    vocab.update();
    clock_t end1 = clock();
    double buildTree_time = double(end1 - begin1) / CLOCKS_PER_SEC;
    cout.precision(5);
    cout<<"buildTree time "<<buildTree_time<<endl;


    cout<<"hahha2 "<<endl;
    vector<KeyPoint> queryKeypoints;
    Mat queryDescriptors;

    ///QueryImage
    {
        string queryImageName="/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000381.ppm";
        Mat queryImg=imread(queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
        detector.detect(queryImg, queryKeypoints);
        extractor.compute(queryImg, queryKeypoints, queryDescriptors);
        cout<<"queryDescriptors.rows: "<<queryDescriptors.rows<<endl;
    }


    Mat indices;
    Mat dists;
    int k=2;

    clock_t begin2 = clock();
    vocab.search(queryDescriptors, indices, dists, k);
    clock_t end2 = clock();
    double query_time = double(end2 - begin2) / CLOCKS_PER_SEC;
    cout.precision(5);
    cout<<"query time "<<query_time<<endl;

    std::vector<int> indicesVec(indices.rows*indices.cols);

    if (indices.isContinuous())
    {
        indicesVec.assign((int*)indices.datastart, (int*)indices.dataend);
    }

    cout<<"indicesVec.size() "<<indicesVec.size()<<endl;

    /// Process Nearest Neighbor Distance Ratio
    float nndRatio = 0.8;

    for(int i=0; i<indicesVec.size(); i++)
    {
        if(dists.at<float>(i,0)<nndRatio*dists.at<float>(i,1))
        {
            cout<<"indicesVec["<<i<<"] "<<indicesVec[i]<<"  image labels "<<labels[indicesVec[i]]<<endl;
        }
    }


    return 0;
}
