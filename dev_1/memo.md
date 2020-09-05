`nn.Linear()` : 첫번째 인자 - input sample의 size, 두번째 인자 - onput sample의 size


```
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
    ```
