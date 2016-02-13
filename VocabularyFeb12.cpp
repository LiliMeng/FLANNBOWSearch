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

void Vocabulary::search(const cv::Mat & descriptors, cv::Mat & results, cv::Mat & dists, int k)
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

void Vocabulary::search_image(int imgNum, vector<int> images, multimap<int, int> &imageScore, vector<int> &rankedScore)
{
    for(int i=0; i<imgNum; i++)
    {
        imageScore.insert(pair<int,int>(i+1,0));
    }

    cout<<imageScore.size()<<endl;

    int score=0;

    for(auto ite=imageScore.begin(); ite!=imageScore.end(); ite++)
    {
        for(int i=0; i<images.size(); i++)
        {
            if((*ite).first==images[i])
            {
                (*ite).second++;
            }

        }

        ///normalize
        float normalized_score=(float)(*ite).second/(float)images.size();

       // (*ite).second=normalized_score;
        cout.precision(5);
        cout<<(*ite).first<<" "<<(*ite).second<<" "<<normalized_score<<endl;
        rankedScore.push_back((*ite).second);
    }
}

