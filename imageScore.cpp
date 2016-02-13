///Practice on imageScore

#include <iostream>
#include <map>
#include <vector>

using namespace std;

void search_image(int imgNum, vector<int> images, multimap<int, float> &imageScore)
{
    for(int i=0; i<imgNum; i++)
    {
        imageScore.insert(pair<int,float>(i+1,0.0));
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

        (*ite).second=normalized_score;
        cout.precision(5);
        cout<<(*ite).first<<" "<<(*ite).second<<" "<<normalized_score<<endl;
    }


}


int main()
{
    multimap<int, float> imageScore;

    int imgNum=50;

    vector<int> images={1,12,13,23,12,13,16,12,18,19};

    search_image(imgNum, images, imageScore);

    return 0;

}
