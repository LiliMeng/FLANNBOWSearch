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

struct RankedScore {
    int imageIndex;
    float imageScore;
};

struct by_number {
    bool operator()(RankedScore const &left, RankedScore const &right) {
        return left.imageScore > right.imageScore;
    }
};


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
	void search_descriptors(const cv::Mat & descriptors, cv::Mat & results, cv::Mat & dists, int k);
	void search_image(const cv::Mat & descriptors, cv::Mat & results, cv::Mat & dists, int k, int imgNum, vector<unsigned int> labels, vector<int> images, multimap<int, int> &imageScore, vector<RankedScore> &rankedScore);

    void computeLikelihood(int imgNum, vector<RankedScore> rankedScore, vector<double> likelihood);

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
