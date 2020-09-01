#include <algorithm>
#include <iostream>
#include <random>
#include <time.h>
#include <vector>

#define NUMBER_OF_SAMPLE 10

using namespace std;

void MakeBlock(vector<int>* label)
{
    static random_device rd;

    vector<int>& rLabel = *label;
    vector<int> result;

    try
    {
        result.resize(NUMBER_OF_SAMPLE);
    }
    catch (...)
    {
        // 예외 처리
        printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__,
               __FILE__, __LINE__);
        return;
    }

    for (size_t i = 0ul; i < NUMBER_OF_SAMPLE; i++)
    {
        result[i] = i;
    }

    // debug
    for (size_t i = 0ul; i < rLabel.size(); i++)
    {
        cout << rLabel[i] << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0ul; i < result.size(); i++)
    {
        cout << result[i] << " ";
    }
    std::cout << std::endl;

    // shuffle
    size_t beginIndex = 0ul; // 셔플할 레이블의 처음 인덱스

    for (size_t i = 1ul; i < rLabel.size(); i++)
    {
        if (rLabel[beginIndex] != rLabel[i])
        {
            shuffle(&result[beginIndex], &result[i - 1ul],
                    default_random_engine(rd()));

            beginIndex = i;
        }
    }

    shuffle(&result[beginIndex], &result[result.size() - 1ul],
            default_random_engine(rd()));

    for (size_t i = 0ul; i < NUMBER_OF_SAMPLE; i++)
    {
        rLabel[i] = result[i];
    }
}

int main()
{
    srand(time(NULL));

    vector<int> label;
    label.push_back(1);
    label.push_back(1);
    label.push_back(1);
    label.push_back(2);
    label.push_back(2);
    label.push_back(2);
    label.push_back(2);
    label.push_back(3);
    label.push_back(4);
    label.push_back(4);
    label.push_back(4);

    MakeBlock(&label);

    for (int i = 0; i < 10; i++)
    {
        std::cout << label[i] << " ";
    }

    std::cout << std::endl;

    return 0;
}
