///Practice on imageScore

#include <iostream>
#include <map>
#include <vector>

using namespace std;




int main()
{
    multimap<int, int> imageScore;

    int imgNum=50;

    for(int i=0; i<imgNum; i++)
    {
        imageScore.insert(pair<int,int>(i+1,0));
    }

    cout<<imageScore.size()<<endl;

    vector<int> image={1,12,13,23,12,13,16,12,18,19};

    int score=0;

    for(auto ite=imageScore.begin(); ite!=imageScore.end(); ite++)
    {
        for(int i=0; i<image.size(); i++)
        {
            if((*ite).first==image[i])
            {
                (*ite).second++;
            }

        }

        ///normalize
        float normalized_score=(float)(*ite).second/(float)image.size();
        cout.precision(5);
        cout<<(*ite).first<<" "<<(*ite).second<<" "<<normalized_score<<endl;
    }

    return 0;

}
