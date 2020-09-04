#include "../../../WICWIU_src/NeuralNetwork.hpp"

template <typename DTYPE>
class Block35 : public Module<DTYPE>
{
private:
public:
    Block35(Operator<DTYPE>* pInput, int pNumInputChannel,
            int pNumOutputChannel, int pStride = 1, std::string pName = NULL)
    {
        Alloc(pInput, pNumInputChannel, pNumOutputChannel, pStride, pName);
    }

    virtual ~Block35() {}

    int Alloc(Operator<DTYPE>* pInput, int pNumInputChannel,
              int pNumOutputChannel, int pStride, std::string pName)
    {
        this->SetInput(pInput);

        Operator<DTYPE>* remember = pInput;
        Operator<DTYPE>* out1 = pInput;
        Operator<DTYPE>* out2 = pInput;
        Operator<DTYPE>* out3 = pInput;
        Operator<DTYPE>* out = pInput;
        Operator<DTYPE>* tmp = pInput;

        // out 1
        out1 = new ConvolutionLayer2D<DTYPE>(
            out1, 256, 32, 1, 1, pStride, pStride, 1, FALSE, "Block35_conv" + pName)
        );
        out1 =
            new BatchNormalizeLayer<DTYPE>(out1, TRUE, "Block35_conv1" + pName);
        out1 = new Relu<DTYPE>(out1, "Block35_Relu1" + pName);

        // out 2-1
        out2 = new ConvolutionLayer2D<DTYPE>(
            out2, 256, 32, 1, 1, pStride, pStride, 1, FALSE, "Block35_conv" + pName)
        );
        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block35_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block35_Relu2" + pName);

        // out 2-2
        out2 =
            new ConvolutionLayer2D<DTYPE>(out2, 32, 32, 3, 3, pStride, pStride,
                                          1, FALSE, "Block35_conv" + pName);
        );
        out2 =
            new BatchNormalizeLayer<DTYPE>(out2, TRUE, "Block35_conv2" + pName);
        out2 = new Relu<DTYPE>(out2, "Block35_Relu2" + pName);

        // out 3-1
        out3 =
            new ConvolutionLayer2D<DTYPE>(out3, 256, 32, 1, 1, pStride, pStride,
                                          1, FALSE, "Block35_conv" + pName);
        );
        out3 =
            new BatchNormalizeLayer<DTYPE>(out3, TRUE, "Block35_conv3" + pName);
        out3 = new Relu<DTYPE>(out3, "Block35_Relu3" + pName);

        out3 =
            new ConvolutionLayer2D<DTYPE>(out3, 32, 32, 3, 3, pStride, pStride,
                                          1, FALSE, "Block35_conv" + pName);
        );
        out3 =
            new BatchNormalizeLayer<DTYPE>(out3, TRUE, "Block35_conv3" + pName);
        out3 = new Relu<DTYPE>(out2, "Block35_Relu3" + pName);

        out3 =
            new ConvolutionLayer2D<DTYPE>(out2, 32, 32, 3, 3, pStride, pStride,
                                          1, FALSE, "Block35_conv" + pName);
        );

        // concat
        out = new ConcatenateChannelWise<DTYPE>(out1, out2, "Block35_ConCat");
        out = new ConcatenateChannelWise<DTYPE>(out3, out, "Block35_ConCat");
        out =
            new ConvolutionLayer2D<DTYPE>(out, 96, 256, 3, 3, pStride, pStride,
                                          1, FALSE, "Block35_conv" + pName);
        );

        // // ShortCut
        // if ((pStride != 1) || (pNumInputChannel != pNumOutputChannel))
        // {
        //     remember = new ConvolutionLayer2D<DTYPE>(
        //         remember, pNumInputChannel, pNumOutputChannel, 1, 1, pStride,
        //         pStride, 0, FALSE, "Block35_Shortcut" + pName);
        //     remember = new BatchNormalizeLayer<DTYPE>(
        //         remember, TRUE, "Block35_Shortcut" + pName);
        // }

        // Add (for skip Connection)
        out = new Addall<DTYPE>(remember, out,
                                "Incpetion_ResNet_Skip_Add" + pName);

        // Last Relu
        out = new Relu<DTYPE>(out, "Block35_Relu" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template <typename DTYPE>
class InceptionResNet : public NeuralNetwork<DTYPE>
{
private:
    int m_numInputChannel;
    int m_numOutputChannel;

public:
    InceptionResNet(Tensorholder<DTYPE>* pInput, Tensorholder<DTYPE>* pLabel,
                    std::string pBlockType, int pNumOfBlock1, int pNumOfBlock2,
                    int pNumOfBlock3, int pNumOfBlock4, int pNumOfClass)
    {
        Alloc(pInput, pLabel, pBlockType, pNumOfBlock1, pNumOfBlock2,
              pNumOfBlock3, pNumOfBlock4, pNumOfClass);
    }

    virtual ~InceptionResNet() {}

    int Alloc(Tensorholder<DTYPE>* pInput, Tensorholder<DTYPE>* pLabel,
              std::string pBlockType, int pNumOfBlock1, int pNumOfBlock2,
              int pNumOfBlock3, int pNumOfBlock4, int pNumOfClass)
    {
        this->SetInput(2, pInput, pLabel);

        // init
        m_numInputChannel = 64;
        m_numOutputChannel = 0;

        Operator<DTYPE>* out = pInput;

        // ReShape
        out = new ReShape<DTYPE>(out, 3, 224, 224, "ReShape");
        // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN0");

        // 1
        out = new ConvolutionLayer2D<DTYPE>(out, 3, m_numInputChannel, 7, 7, 2,
                                            2, 3, FALSE, "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN0");
        out = new Relu<DTYPE>(out, "Relu0");

        out = new Maxpooling2D<float>(out, 3, 3, 2, 2, 1, "MaxPool_2");
        // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1");

        out = this->MakeLayer(out, m_numInputChannel, pBlockType, pNumOfBlock1,
                              1, "Block1");
        out = this->MakeLayer(out, 128, pBlockType, pNumOfBlock2, 2, "Block2");
        out = this->MakeLayer(out, 256, pBlockType, pNumOfBlock3, 2, "Block3");
        out = this->MakeLayer(out, 512, pBlockType, pNumOfBlock3, 2, "Block4");

        // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1");
        // out = new Relu<DTYPE>(out, "Relu1");

        out = new GlobalAvaragePooling2D<DTYPE>(out, "Avg Pooling");

        out = new ReShape<DTYPE>(out, 1, 1, 512, "ReShape");

        out = new Linear<DTYPE>(out, 512, pNumOfClass, FALSE, "Classification");
        // out = new BatchNormalizeLayer < DTYPE > (out, FALSE, "BN0");

        this->AnalyzeGraph(out);

        // ======================= Select LossFunction Function
        // ===================
        this->SetLossFunction(
            new SoftmaxCrossEntropy<float>(out, pLabel, "SCE"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================

        this->SetOptimizer(new AdamOptimizer<float>(
            this->GetParameter(), 0.001, 0.9, 0.999, 1e-08, 5e-4, MINIMIZE));

        return TRUE;
    }

    Operator<DTYPE>* MakeLayer(Operator<DTYPE>* pInput, int pNumOfChannel,
                               std::string pBlockType, int pNumOfBlock,
                               int pStride, std::string pName = NULL)
    {
        if (pNumOfBlock == 0)
        {
            return pInput;
        }
        else if ((pBlockType == "BasicBlock") && (pNumOfBlock > 0))
        {
            Operator<DTYPE>* out = pInput;

            out = new BasicBlock<DTYPE>(out, m_numInputChannel, pNumOfChannel,
                                        pStride, pName);
            int pNumOutputChannel = pNumOfChannel;

            for (int i = 1; i < pNumOfBlock; i++)
            {
                out = new BasicBlock<DTYPE>(out, pNumOutputChannel,
                                            pNumOutputChannel, 1, pName);
            }

            m_numInputChannel = pNumOutputChannel;

            return out;
        }
        else if ((pBlockType == "Bottleneck") && (pNumOfBlock > 0))
        {
            return NULL;
        }
        else
            return NULL;
    }
};

template <typename DTYPE>
NeuralNetwork<DTYPE>* Resnet18(Tensorholder<DTYPE>* pInput,
                               Tensorholder<DTYPE>* pLabel, int pNumOfClass)
{
    return new ResNet<DTYPE>(pInput, pLabel, "BasicBlock", 2, 2, 2, 2,
                             pNumOfClass);
}

template <typename DTYPE>
NeuralNetwork<DTYPE>* Resnet34(Tensorholder<DTYPE>* pInput,
                               Tensorholder<DTYPE>* pLabel, int pNumOfClass)
{
    return new ResNet<DTYPE>(pInput, pLabel, "BasicBlock", 3, 4, 6, 3,
                             pNumOfClass);
}
