#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <cassert>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography

using namespace std;


class Vocabulary
{
    public:
        Vocabulary(int imgNum);
        virtual ~Vocabulary();
    void clear();
    multimap<int, int> addWords(const cv::Mat & descriptors, int objectIndex, bool incremenal);
	void update();
	void search(const cv::Mat & descriptors, cv::Mat & results, cv::Mat & dists, int k);
	int size() const
	{
        return indexedDescriptors_.rows + notIndexedDescriptors_.rows;
	}

	const multimap<int, int> & wordToObjects() const
	{
        return wordToObjects_;
	}

	int clustering(const cv::Mat& features, cv::Mat& centers);
	void search_image(int imgNum, vector<int> images, multimap<int, int> &imageScore, vector<int> &rankedScore);

    void queryDistribution(const cv::Mat & features, vector<float> & distribution) const;
    void computeLikelihood(const cv::Mat & features, vector<float> &likelihood) const;

    cv::flann::SearchParams getFlannSearchParams();
    cvflann::flann_distance_t getFlannDistanceType();
    static cv::flann::IndexParams * createFlannIndexParams(int index);

    int nLabel_;

    cv::flann::Index flannIndex_;
    cv::Mat indexedDescriptors_;
    cv::Mat notIndexedDescriptors_;
    multimap<int, int> wordToObjects_;
    vector<int> notIndexedWordIds_;
};

#endif // VOCABULARY_H
