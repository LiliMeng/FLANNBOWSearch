#include "Vocabulary.h"

#include "opencv2/flann/miniflann.hpp"
#include "opencv2/flann/flann_base.hpp"
#include "opencv2/flann/flann.hpp"

Vocabulary::Vocabulary(int imgNum)
{
    nLabel_=imgNum;
}

Vocabulary::~Vocabulary()
{
    //dtor
}

void Vocabulary::clear()
{
    indexedDescriptors_ = cv::Mat();
    notIndexedDescriptors_ = cv::Mat();
    wordToObjects_.clear();
    notIndexedWordIds_.clear();

}

cv::flann::SearchParams Vocabulary::getFlannSearchParams()
{
    int checks=32;
    float eps=0;
    bool sorted=true;
	return cv::flann::SearchParams(
			32, 0, true);
}

cvflann::flann_distance_t Vocabulary::getFlannDistanceType()
{
	cvflann::flann_distance_t distance = cvflann::FLANN_DIST_L2;

	return distance;
}

cv::flann::IndexParams * Vocabulary::createFlannIndexParams(int index)
{
	cv::flann::IndexParams * params = 0;

    switch(index)
    {
        case 0:
            params = new cv::flann::LinearIndexParams();
            cout<<"flann linearIndexParams"<<endl;
            break;
        case 1:
            params = new cv::flann::KDTreeIndexParams();
            cout<<"flann randomized kdtrees params"<<endl;
            break;
        case 2:
            params = new cv::flann::KMeansIndexParams();
            cout<<"flann hierarchical k-means tree params"<<endl;
            break;
        case 3:
            params = new cv::flann::CompositeIndexParams();
            cout<<"flann combination of randomized kd-trees and hierarchical k-means tree"<<endl;
            break;
        /*case 4:
            params = new cv::flann::LshIndexParams();
            cout<<"flann LSH Index params"<<endl;
            break;*/
        case 5:
            params = new cv::flann::AutotunedIndexParams();
            cout<<"flann AutotunedIndexParams"<<endl;
            break;
        default:
            break;
    }

	if(!params)
	{
		printf("ERROR: NN strategy not found !? Using default KDTRee...\n");
		params = new cv::flann::KDTreeIndexParams();
	}
	return params ;
}

int Vocabulary::clustering(const cv::Mat& features, cv::Mat& centers)
{

    assert(features.type()==CV_32F);
    assert(centers.type()==CV_32F);
    int number_of_clusters;
    assert(features.isContinuous());

    cvflann::Matrix<float> featuresFlann((float*)features.data, features.rows, features.cols);
    cvflann::Matrix<float> centersFlann((float*)centers.data, centers.rows, centers.cols);
    int branching=10;
    int iterations=15;
    cvflann::flann_centers_init_t centers_init= cvflann::FLANN_CENTERS_GONZALES;
    float cb_index=0.2f;

    cvflann::KMeansIndexParams  k_params=cvflann::KMeansIndexParams(branching,iterations,centers_init, cb_index);

    //cv::flann::KMeansIndexParams k_params(10, 1000, cvflann::FLANN_CENTERS_KMEANSPP,0.01);

    number_of_clusters= cvflann::hierarchicalClustering<cvflann::L2<float> >(featuresFlann,centersFlann,k_params);

    return number_of_clusters;

}
void Vocabulary::update()
{
	if(!notIndexedDescriptors_.empty())
	{
		assert(indexedDescriptors_.cols == notIndexedDescriptors_.cols &&
				 indexedDescriptors_.type() == notIndexedDescriptors_.type());

		//concatenate descriptors
		indexedDescriptors_.push_back(notIndexedDescriptors_);  //插入新进来的图片的IndexedDescriptors

		notIndexedDescriptors_ = cv::Mat();
		notIndexedWordIds_.clear(); //把新加进来的notIndexedwords清空
	}

	if(!indexedDescriptors_.empty())
	{
		cv::flann::IndexParams * params = createFlannIndexParams(1);
		flannIndex_.build(indexedDescriptors_, *params, getFlannDistanceType());
		delete params;
	}
}

void Vocabulary::search_descriptors(const cv::Mat & descriptors, cv::Mat & results, cv::Mat & dists, int k)
{
	assert(notIndexedDescriptors_.empty() && notIndexedWordIds_.size() == 0);

	if(!indexedDescriptors_.empty())
	{
		assert(descriptors.type() == indexedDescriptors_.type() && descriptors.cols == indexedDescriptors_.cols);

		flannIndex_.knnSearch(descriptors, results, dists, k, getFlannSearchParams());

		if( dists.type() == CV_32S )
		{
			cv::Mat temp;
			dists.convertTo(temp, CV_32F);
			dists = temp;
		}
	}
}

void Vocabulary::search_image(const cv::Mat & queryDescriptors, cv::Mat & indices, cv::Mat & dists, int k, int imgNum, vector<unsigned int> labels, vector<int> images, multimap<int, int> &imageScore, vector<RankedScore> &rankedScore)
{
  //  clock_t begin2 = clock();
    this->search_descriptors(queryDescriptors, indices, dists, k);
    //clock_t end2 = clock();
    //double query_time = double(end2 - begin2) / CLOCKS_PER_SEC;
    cout.precision(5);
   // cout<<"query time "<<query_time<<endl;

    std::vector<int> indicesVec(indices.rows*indices.cols);

    if (indices.isContinuous())
    {
        indicesVec.assign((int*)indices.datastart, (int*)indices.dataend);
    }

    cout<<"indicesVec.size() "<<indicesVec.size()<<endl;

    vector<int> imageLabelsVec;

    /// Process Nearest Neighbor Distance Ratio
    float nndRatio = 0.8;

    for(int i=0; i<indicesVec.size(); i++)
    {
       // if(dists.at<float>(i,0)<nndRatio*dists.at<float>(i,1))
       // {
           // cout<<"indicesVec["<<i<<"] "<<indicesVec[i]<<"  image labels "<<labels[indicesVec[i]]<<endl;
            imageLabelsVec.push_back(labels[indicesVec[i]]);
        //}
    }



    for(int i=0; i<imgNum; i++)
    {
        imageScore.insert(pair<int,int>(i+1,0));
    }

    cout<<imageScore.size()<<endl;

    int score=0;

    for(auto ite=imageScore.begin(); ite!=imageScore.end(); ite++)
    {
        for(int i=0; i<imageLabelsVec.size(); i++)
        {
            if((*ite).first==imageLabelsVec[i])
            {
                (*ite).second++;
            }

        }

        ///normalize
        float normalized_score=(float)(*ite).second/(float)imageLabelsVec.size();

        RankedScore singleRankedScore;

       // (*ite).second=normalized_score;
       // cout.precision(5);
        //cout<<(*ite).first<<" "<<(*ite).second<<" "<<normalized_score<<endl;
        singleRankedScore.imageIndex=(*ite).first;
        singleRankedScore.imageScore=normalized_score;
        rankedScore.push_back(singleRankedScore);
    }


        sort(rankedScore.begin(),rankedScore.end(),by_number());

        cout.precision(10);
        for(int i=0; i<rankedScore.size();i++)
        {
            cout<<rankedScore[i].imageIndex<<" "<<rankedScore[i].imageScore<<endl;
        }
}

