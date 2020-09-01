#include <algorithm>
#include <iostream>
#include <random>
#include <time.h>
#include <vector>

#define NUMBER_OF_SAMPLE 10

using namespace std;

void MakeBlock(vector<vector<int>>* label, vector<int>* result)
{

    static random_device rd;

    try
    {
        result->resize(NUMBER_OF_SAMPLE);
        result->clear();
    }
    catch (...)
    {
        printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__,
               __FILE__, __LINE__);
        return;
    }

    // debug
    for (unsigned int i = 0; i < (*label).size(); i++)
    {
        if ((*label)[i].empty() == true)
            continue;

        for (int j = 0; j < (*label)[i].size(); j++)
        {
            cout << (*label)[i][j] << " ";
        }
    }
    std::cout << std::endl;

    for (unsigned int i = 0; i < (*label).size(); i++)
    {
        if ((*label)[i].empty() == true)
            continue;

        for (int j = 0; j < (*label)[i].size(); j++)
        {
            cout << i << " ";
        }
    }
    std::cout << std::endl;

    // shuffle
    for (unsigned int i = 0; i < (*label).size(); i++)
    {
        if ((*label)[i].empty() == true)
            continue;

        shuffle((*label)[i].begin(), (*label)[i].end(),
                default_random_engine(rd()));
    }

    for (unsigned int i = 0; i < (*label).size(); i++)
    {
        if ((*label)[i].empty() == true)
            continue;

        for (unsigned int j = 0; j < (*label)[i].size(); j++)
        {
            result->push_back((*label)[i][j]);
        }
    }
}

int main()
{
    srand(time(NULL));
    vector<vector<int>> label(5);
    vector<int> result(NUMBER_OF_SAMPLE);

    label[1].push_back(0);
    label[1].push_back(1);
    label[1].push_back(2);
    label[2].push_back(3);
    label[2].push_back(4);
    label[2].push_back(5);
    label[2].push_back(6);
    label[3].push_back(7);
    label[4].push_back(8);
    label[4].push_back(9);
    MakeBlock(&label, &result);

    for (int i = 0; i < 10; i++)
    {
        std::cout << result[i] << " ";
    }

    std::cout << std::endl;

    return 0;
}
