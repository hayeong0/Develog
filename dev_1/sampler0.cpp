#include <iostream>
#include <time.h>
#include <vector>
#include <algorithm>

#define NUMBER_OF_SAMPLE 10

std::vector<int> MakeBlock(std::vector<int> *label) {
   int maxLabel = (*label).back();
   std::vector<int> shuffle;
   std::vector<int> labelShuffle;
   std::vector<int> orderedLabel;
   std::vector<int> result;

   try {
      shuffle.resize(NUMBER_OF_SAMPLE);
      labelShuffle.resize(NUMBER_OF_SAMPLE);
      orderedLabel.resize(NUMBER_OF_SAMPLE);
   }
   catch (...) {
      printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
      return result;
   }

   // initialize
   for (int i = 0; i < NUMBER_OF_SAMPLE; i++) {
      shuffle[i] = i;
      labelShuffle[i] = orderedLabel[i] = (*label)[i];
   }

   // shuffling and copy
   int sour;
   int dest;
   int temp;

   for (int i = 0, range = NUMBER_OF_SAMPLE * 10; i < range; i++) {
      sour = rand() % NUMBER_OF_SAMPLE;
      dest = rand() % NUMBER_OF_SAMPLE;

      temp = shuffle[dest];
      shuffle[dest] = shuffle[sour];
      shuffle[sour] = temp;

      temp = labelShuffle[dest];
      labelShuffle[dest] = labelShuffle[sour];
      labelShuffle[sour] = temp;
   }


   std::sort(orderedLabel.begin(), orderedLabel.end(), std::less<int>());

   // for debug
   std::cout << "shuffle vector : ";
   for (int i = 0; i < NUMBER_OF_SAMPLE; i++) {
      std::cout << shuffle[i] << " ";
   }
   std::cout << std::endl;

    std::cout << "label vector   : ";
   for (int i = 0; i < NUMBER_OF_SAMPLE; i++) {
      std::cout << (*label)[i] << " ";
   }
   std::cout << std::endl;

   label->resize(NUMBER_OF_SAMPLE);

   int tempLabel = orderedLabel[0];

   // first exception
   for (int i = 0; i < NUMBER_OF_SAMPLE; i++) {
      if (labelShuffle[i] == tempLabel)
         result.push_back(shuffle[i]);
   }

   // result calc
   for (int i = 0; i < NUMBER_OF_SAMPLE; i++) {
      if (tempLabel == orderedLabel[i]) continue;
      else tempLabel = orderedLabel[i];

      for (int j = 0; j < NUMBER_OF_SAMPLE; j++) {
         if (labelShuffle[j] == tempLabel)
            result.push_back(shuffle[j]);
      }
   }

   return result;
}

int main() {
   srand(time(NULL));
   std::vector<int> label;
   std::vector<int> result;

   // for testing
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
   // label.push_back(1);
   // label.push_back(1);
   // label.push_back(1);
   // label.push_back(2);
   // label.push_back(2);
   // label.push_back(2);
   // label.push_back(2);
   // label.push_back(3);
   // label.push_back(4);
   // label.push_back(4);
   result = MakeBlock(&label);

    std::cout << "Indexed        : ";
   for (int i = 0; i < NUMBER_OF_SAMPLE; i++) {
      std::cout << result[i] << " " ;
   }
   std::cout << std::endl;

   return 0;
} 
