#include "WICWIU/WICWIU_src/NeuralNetwork.hpp"
#include "WICWIU/WICWIU_src/Operator/Dropout.hpp"
// #include <cuda.h>
#include <iostream>
#include <string>

template <typename DTYPE>
class Block35 : public Module<DTYPE>
{
private:
public:
    Block35(Operator<DTYPE>* pInput, int pStride = 1, std::string pName = NULL)
        : Module<DTYPE>(pName)
    {
        Alloc(pInput, pStride, pName);
    }

    virtual ~Block35() {}

    int Alloc(Operator<DTYPE>* pInput, int pStride, std::string pName)
    {
        this->SetInput(pInput);

        Operator<DTYPE>* remember = pInput;
        Operator<DTYPE>* out1 = pInput;
        Operator<DTYPE>* out2 = pInput;
        Operator<DTYPE>* out3 = pInput;
        Operator<DTYPE>* out = pInput;
        Operator<DTYPE>* tmp = pInput;

        // out 1
        out1 =
            new ConvolutionLayer2D<DTYPE>(out1, 256, 32, 1, 1, pStride, pStride,
                                          0, FALSE, "Block35_conv" + pName);
        out1 =
            new BatchNormalizeLayer<DTYPE>(out1, TRUE, "Block35_conv1" + pName);
        out1 = new Relu<DTYPE>(out1, "Block35_Relu1" + pName);

        // out 2-1
        out2 =
            new ConvolutionLayer2D<DTYPE>(out2, 256, 32, 1, 1, pStride, pStride,
                                          0, FALSE, "Block35_conv" + pName);

        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block35_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block35_Relu2" + pName);

        // out 2-2
        out2 =
            new ConvolutionLayer2D<DTYPE>(out2, 32, 32, 3, 3, pStride, pStride,
                                          1, FALSE, "Block35_conv" + pName);

        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block35_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block35_Relu2" + pName);

        // out 3-1
        out3 =
            new ConvolutionLayer2D<DTYPE>(out3, 256, 32, 1, 1, pStride, pStride,
                                          0, FALSE, "Block35_conv" + pName);
        out3 =
            new BatchNormalizeLayer<DTYPE>(out3, TRUE, "Block35_conv3" + pName);
        out3 = new Relu<DTYPE>(out3, "Block35_Relu3" + pName);
        
        // out 3-2
        out3 =
            new ConvolutionLayer2D<DTYPE>(out3, 32, 32, 3, 3, pStride, pStride,
                                          1, FALSE, "Block35_conv" + pName);
        out3 =
            new BatchNormalizeLayer<DTYPE>(out3, TRUE, "Block35_conv3" + pName);
        out3 = new Relu<DTYPE>(out3, "Block35_Relu3" + pName);

        // out 3-3
        out3 =
            new ConvolutionLayer2D<DTYPE>(out3, 32, 32, 3, 3, pStride, pStride,
                                          1, FALSE, "Block35_conv" + pName);
        out3 =
            new BatchNormalizeLayer<DTYPE>(out3, TRUE, "Block35_conv3" + pName);
        out3 = new Relu<DTYPE>(out3, "Block35_Relu3" + pName);

        // concat
        out = new ConcatenateChannelWise<DTYPE>(out1, out2, "Block35_ConCat");
        out = new ConcatenateChannelWise<DTYPE>(out3, out, "Block35_ConCat");
        out =
            new ConvolutionLayer2D<DTYPE>(out, 96, 256, 3, 3, pStride, pStride,
                                          0, FALSE, "Block35_conv" + pName);

        // Add (for skip Connection)
        out = new Addall<DTYPE>(remember, out,
                                "Incpetion_ResNet_Skip_Add " + pName);

        // Last Relu
        out = new Relu<DTYPE>(out, "Block35_Relu" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template <typename DTYPE>
class Block17 : public Module<DTYPE>
{
private:
public:
    Block17(Operator<DTYPE>* pInput, int pStride = 1, std::string pName = NULL)
        : Module<DTYPE>(pName)
    {
        Alloc(pInput, pStride, pName);
    }

    virtual ~Block17() {}

    int Alloc(Operator<DTYPE>* pInput, int pStride, std::string pName)
    {
        this->SetInput(pInput);

        Operator<DTYPE>* remember = pInput;
        Operator<DTYPE>* out1 = pInput;
        Operator<DTYPE>* out2 = pInput;
        Operator<DTYPE>* out = pInput;

        // out 1
        out1 = new ConvolutionLayer2D<DTYPE>(out1, 896, 128, 1, 1, pStride,
                                             pStride, 0, FALSE,
                                             "Block17_conv" + pName);
        out1 =
            new BatchNormalizeLayer<DTYPE>(out1, TRUE, "Block17_conv1" + pName);
        out1 = new Relu<DTYPE>(out1, "Block17_Relu1" + pName);

        // out 2-1
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 896, 128, 1, 1, pStride,
                                             pStride, 0, FALSE,
                                             "Block17_conv" + pName);

        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block17_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block17_Relu2" + pName);

        // out 2-2
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 128, 128, 1, 7, pStride,
                                             pStride, 0, 3, FALSE,
                                             "Block17_conv" + pName);
        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block17_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block17_Relu2" + pName);

        // out 2-3
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 128, 128, 7, 1, pStride,
                                             pStride, 3, 0, FALSE,
                                             "Block17_conv" + pName);
        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block17_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block17_Relu2" + pName);

        // concat
        out = new ConcatenateChannelWise<DTYPE>(out1, out2, "Block17_ConCat");
        out =
            new ConvolutionLayer2D<DTYPE>(out, 256, 896, 1, 1, pStride, pStride,
                                          0, FALSE, "Block17_conv" + pName);

        // Add (for skip Connection)
        out = new Addall<DTYPE>(remember, out,
                                "Incpetion_ResNet_Skip_Add" + pName);

        // Last Relu
        out = new Relu<DTYPE>(out, "Block17_Relu" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template <typename DTYPE>
class Block8 : public Module<DTYPE>
{
private:
public:
    Block8(Operator<DTYPE>* pInput, int pStride = 1, std::string pName = NULL)
        : Module<DTYPE>(pName)
    {
        Alloc(pInput, pStride, pName);
    }

    virtual ~Block8() {}

    int Alloc(Operator<DTYPE>* pInput, int pStride, std::string pName)
    {
        this->SetInput(pInput);

        Operator<DTYPE>* remember = pInput;
        Operator<DTYPE>* out1 = pInput;
        Operator<DTYPE>* out2 = pInput;
        Operator<DTYPE>* out = pInput;

        // out 1
        out1 = new ConvolutionLayer2D<DTYPE>(out1, 1792, 192, 1, 1, pStride,
                                             pStride, 0, FALSE,
                                             "Block8_conv" + pName);
        out1 =
            new BatchNormalizeLayer<DTYPE>(out1, TRUE, "Block8_conv1" + pName);
        out1 = new Relu<DTYPE>(out1, "Block8_Relu1" + pName);

        // out 2-1
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 1792, 192, 1, 1, pStride,
                                             pStride, 0, FALSE,
                                             "Block8_conv" + pName);
        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block8_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block8_Relu2" + pName);

        // out 2-2
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 192, 192, 1, 3, pStride,
                                             pStride, 0, 1, FALSE,
                                             "Block8_conv" + pName);
        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block8_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block8_Relu2" + pName);

        // out 2-3
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 192, 192, 3, 1, pStride,
                                             pStride, 1, 0, FALSE,
                                             "Block8_conv" + pName);
        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block8_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block8_Relu2" + pName);

        // concat
        out = new ConcatenateChannelWise<DTYPE>(out1, out2, "Block8_ConCat");
        out = new ConvolutionLayer2D<DTYPE>(out, 384, 1792, 1, 1, pStride,
                                            pStride, 0, FALSE,
                                            "Block8_conv" + pName);

        // Add (for skip Connection)
        out = new Addall<DTYPE>(remember, out,
                                "Incpetion_ResNet_Skip_Add" + pName);

        // Last Relu
        out = new Relu<DTYPE>(out, "Block8_Relu" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};
template <typename DTYPE>
class ReductionA : public Module<DTYPE>
{
private:
public:
    ReductionA(Operator<DTYPE>* pInput, int pStride = 1,
               std::string pName = NULL)
        : Module<DTYPE>(pName)
    {
        Alloc(pInput, pStride, pName);
    }

    virtual ~ReductionA() {}

    int Alloc(Operator<DTYPE>* pInput, int pStride, std::string pName)
    {
        this->SetInput(pInput);

        Operator<DTYPE>* remember = pInput;
        Operator<DTYPE>* out1 = pInput;
        Operator<DTYPE>* out2 = pInput;
        Operator<DTYPE>* out3 = pInput;
        Operator<DTYPE>* out = pInput;

        // out 1
        out1 = new ConvolutionLayer2D<DTYPE>(out1, 256, 384, 3, 3, 2, 2, 0,
                                             FALSE, "ReductionA" + pName);
        out1 = new BatchNormalizeLayer<DTYPE>(out1, TRUE, "ReductionA" + pName);
        out1 = new Relu<DTYPE>(out1, "ReductionA_Relu1" + pName);

        // out 2-1
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 256, 192, 1, 1, pStride,
                                             pStride, 0, FALSE,
                                             "ReductionA" + pName);
        out2 = new BatchNormalizeLayer<DTYPE>(out2, TRUE, "ReductionA" + pName);
        out2 = new Relu<DTYPE>(out2, "ReductionA_Relu2-1" + pName);

        // out 2-2
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 192, 192, 3, 3, pStride,
                                             pStride, 1, FALSE,
                                             "ReductionA" + pName);
        out2 = new BatchNormalizeLayer<DTYPE>(out2, TRUE, "ReductionA" + pName);
        out2 = new Relu<DTYPE>(out2, "ReductionA_Relu2-2" + pName);

        // out 2-3
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 192, 256, 3, 3, 2, 2, 0,
                                             FALSE, "ReductionA" + pName);
        out2 = new BatchNormalizeLayer<DTYPE>(out2, TRUE, "ReductionA" + pName);
        out2 = new Relu<DTYPE>(out2, "ReductionA_Relu2-3" + pName);

        // out3
        out3 = new Maxpooling2D<float>(out3, 3, 3, 2, 2, "MaxPool");

        // concat
        out =
            new ConcatenateChannelWise<DTYPE>(out1, out2, "ReductionA_ConCat");
        out = new ConcatenateChannelWise<DTYPE>(out, out3, "ReductionA_ConCat");

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template <typename DTYPE>
class ReductionB : public Module<DTYPE>
{
private:
public:
    ReductionB(Operator<DTYPE>* pInput, int pStride = 1,
               std::string pName = NULL)
        : Module<DTYPE>(pName)
    {
        Alloc(pInput, pStride, pName);
    }

    virtual ~ReductionB() {}

    int Alloc(Operator<DTYPE>* pInput, int pStride, std::string pName)
    {
        this->SetInput(pInput);

        Operator<DTYPE>* remember = pInput;
        Operator<DTYPE>* out1 = pInput;
        Operator<DTYPE>* out2 = pInput;
        Operator<DTYPE>* out3 = pInput;
        Operator<DTYPE>* out4 = pInput;
        Operator<DTYPE>* out = pInput;

        // out 1
        out1 = new ConvolutionLayer2D<DTYPE>(out1, 896, 256, 1, 1, pStride,
                                             pStride, 0, FALSE,
                                             "ReductionB" + pName);
        out1 = new BatchNormalizeLayer<DTYPE>(out1, TRUE, "ReductionB" + pName);
        out1 = new Relu<DTYPE>(out1, "ReductionB_Relu1" + pName);

        // out 1-2
        out1 = new ConvolutionLayer2D<DTYPE>(out1, 256, 384, 3, 3, 2, 2, 0,
                                             FALSE, "ReductionB" + pName);
        out1 = new BatchNormalizeLayer<DTYPE>(out1, TRUE, "ReductionB" + pName);
        out1 = new Relu<DTYPE>(out1, "ReductionB_Relu2" + pName);

        // out 2-1
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 896, 256, 1, 1, pStride,
                                             pStride, 0, FALSE,
                                             "ReductionB" + pName);
        out2 = new BatchNormalizeLayer<DTYPE>(out2, TRUE, "ReductionB" + pName);
        out2 = new Relu<DTYPE>(out2, "ReductionB_Relu2" + pName);

        // out 2-2
        out2 = new ConvolutionLayer2D<DTYPE>(out2, 256, 256, 3, 3, 2, 2, 0,
                                             FALSE, "ReductionB" + pName);
        out2 = new BatchNormalizeLayer<DTYPE>(out2, TRUE, "ReductionB" + pName);
        out2 = new Relu<DTYPE>(out2, "ReductionB_Relu2-2" + pName);

        // out 3-1
        out3 = new ConvolutionLayer2D<DTYPE>(out3, 896, 256, 1, 1, pStride,
                                             pStride, 0, FALSE,
                                             "ReductionB" + pName);
        out3 = new BatchNormalizeLayer<DTYPE>(out3, TRUE, "ReductionB" + pName);
        out3 = new Relu<DTYPE>(out3, "ReductionB_Relu3" + pName);

        // out 3-2
        out3 = new ConvolutionLayer2D<DTYPE>(out3, 256, 256, 3, 3, pStride,
                                             pStride, 1, FALSE,
                                             "ReductionB" + pName);
        out3 = new BatchNormalizeLayer<DTYPE>(out3, TRUE, "ReductionB" + pName);
        out3 = new Relu<DTYPE>(out3, "ReductionB_Relu3" + pName);

        // out 3-3
        out3 = new ConvolutionLayer2D<DTYPE>(out3, 256, 256, 3, 3, 2, 2, 0,
                                             FALSE, "ReductionB" + pName);
        out3 = new BatchNormalizeLayer<DTYPE>(out3, TRUE, "ReductionB" + pName);
        out3 = new Relu<DTYPE>(out3, "ReductionB_Relu3" + pName);

        // Out 4
        out4 = new Maxpooling2D<DTYPE>(out4, 3, 3, 2, 2, "MaxPool");

        // concat
        out =
            new ConcatenateChannelWise<DTYPE>(out1, out2, "ReductionB_ConCat");
        out = new ConcatenateChannelWise<DTYPE>(out, out3, "ReductionB_ConCat");
        out = new ConcatenateChannelWise<DTYPE>(out, out4, "ReductionB_ConCat");

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template <typename DTYPE>
class InceptionResNet : public NeuralNetwork<DTYPE>
{
private:

public:
    InceptionResNet(Tensorholder<DTYPE>* pInput, Tensorholder<DTYPE>* pLabel,
                    int pNumOfClass)
    {
        Alloc(pInput, pLabel, pNumOfClass);
    }

    virtual ~InceptionResNet() {}

    int Alloc(Tensorholder<DTYPE>* pInput, Tensorholder<DTYPE>* pLabel,
              int pNumOfClass)
    {
        this->SetInput(2, pInput, pLabel);

        Operator<DTYPE>* out = pInput;

        /* stem  layer */

        // ReShape
        out = new ReShape<DTYPE>(out, 3, 160, 160, "ReShape");

        // 1
        out = new ConvolutionLayer2D<DTYPE>(out, 3, 32, 3, 3, 2, 2, 0, TRUE,
                                            "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN");
        out = new Relu<DTYPE>(out, "Relu");

        printf("Testing 1\n");
        // 2
        out = new ConvolutionLayer2D<DTYPE>(out, 32, 48, 3, 3, 1, 1, 0, TRUE,
                                            "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN");
        out = new Relu<DTYPE>(out, "Relu");

        // 3
        out = new ConvolutionLayer2D<DTYPE>(out, 48, 64, 3, 3, 1, 1, 1, TRUE,
                                            "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN");
        out = new Relu<DTYPE>(out, "Relu");

        // 4
        out = new Maxpooling2D<DTYPE>(out, 3, 3, 2, 2, 0, "MaxPool");

        // 5
        out = new ConvolutionLayer2D<DTYPE>(out, 64, 80, 1, 1, 1, 1, 0, TRUE,
                                            "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN");
        out = new Relu<DTYPE>(out, "Relu");

        // 6
        out = new ConvolutionLayer2D<DTYPE>(out, 80, 192, 3, 3, 1, 1, 0, TRUE,
                                            "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN");
        out = new Relu<DTYPE>(out, "Relu");

        // 7
        out = new ConvolutionLayer2D<DTYPE>(out, 192, 256, 3, 3, 2, 2, 0,TRUE,
                                            "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN");
        out = new Relu<DTYPE>(out, "Relu");

        printf("Testing 2\n");
        /* end stem  layer */

        // Block35 * 5
        out = new Block35<DTYPE>(out, 1, "Block35_1");
        out = new Block35<DTYPE>(out, 1, "Block35_2");
        out = new Block35<DTYPE>(out, 1, "Block35_3");
        out = new Block35<DTYPE>(out, 1, "Block35_4");
        out = new Block35<DTYPE>(out, 1, "Block35_5");


        printf("Testing 3\n");
        // ReductionA
        out = new ReductionA<DTYPE>(out, 1, "ReductionA");

        // Block17 * 10
        out = new Block17<DTYPE>(out, 1, "Block17_1");
        out = new Block17<DTYPE>(out, 1, "Block17_2");
        out = new Block17<DTYPE>(out, 1, "Block17_3");
        out = new Block17<DTYPE>(out, 1, "Block17_4");
        out = new Block17<DTYPE>(out, 1, "Block17_5");
        out = new Block17<DTYPE>(out, 1, "Block17_6");
        out = new Block17<DTYPE>(out, 1, "Block17_7");
        out = new Block17<DTYPE>(out, 1, "Block17_8");
        out = new Block17<DTYPE>(out, 1, "Block17_9");
        out = new Block17<DTYPE>(out, 1, "Block17_10");

        // ReductionB
        out = new ReductionB<DTYPE>(out, 1, "ReductionB");

        // Block8 * 5
        out = new Block8<DTYPE>(out, 1, "Block8_1");
        out = new Block8<DTYPE>(out, 1, "Block8_2");
        out = new Block8<DTYPE>(out, 1, "Block8_3");
        out = new Block8<DTYPE>(out, 1, "Block8_4");
        out = new Block8<DTYPE>(out, 1, "Block8_5");

        // Average pooling
        out = new GlobalAvaragePooling2D<float>(out, "Avg Pooling");
        printf("Testing 4\n");
        // Dropout
        //out = new Dropout<DTYPE>(out, 0.8, "Dropout");
        out = new ReShape<DTYPE>(out, 1792, 1, 1, "ReShape");
        out = new Linear<DTYPE>(out, 1792, pNumOfClass, FALSE, "Classification");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN");
        printf("Testing 5\n");
        this->AnalyzeGraph(out);
        printf("After analyzegraph\n");
        // softmax
        this->SetLossFunction(
            new SoftmaxCrossEntropy<float>(out, pLabel, "SCE"));
        printf("before optimizer\n");
        // ======================= Select Optimizer ===================
        this->SetOptimizer(new AdamOptimizer<float>(
            this->GetParameter(), 0.001, 0.9, 0.999, 1e-08, 5e-4, MINIMIZE));
         printf("After optimizer \n");
        // this->SetOptimizer(new AdamOptimizer<float>(
        //     this->GetParameter(), 0.0001, 0.9, 0.999, 1e-08, MINIMIZE));
        // this->SetOptimizer(new AdamOptimizer<float>(
        //     this->GetParameter(), 0.00001, 0.9, 0.999, 1e-08, MINIMIZE));
        
        return TRUE;
    }
};
