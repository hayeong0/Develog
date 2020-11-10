#ifndef QUADRUPLETLOSS_H_
#define QUADRUPLETLOSS_H_ value

#include "./WICWIU/WICWIU_src/LossFunction.hpp"

template <typename DTYPE>
class QuadrupletLoss : public LossFunction<DTYPE>
{
private:
    Operator<DTYPE>* m_alpha;
    Operator<DTYPE>* m_beta;
    DTYPE** m_sampleLoss;
    int m_numAnchor;

public:
    QuadrupletLoss(Operator<DTYPE>* pOperator, Operator<DTYPE>* alpha, Operator<DTYPE>* beta,
                   std::string pName = "NO NAME")
        : LossFunction<DTYPE>(pOperator, NULL, pName)
    {
        Alloc(pOperator, alpha, beta);
    }

    ~QuadrupletLoss()
    {
        Delete();
    }

    int Alloc(Operator<DTYPE>* pOperator, Operator<DTYPE>* alpha, Operator<DTYPE>* beta)
    {
        Operator<DTYPE>* pInput = pOperator;

        int timesize = pInput->GetResult()->GetTimeSize();
        int batchSize = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize = pInput->GetResult()->GetRowSize();
        m_alpha = alpha;
        m_beta = beta;
        m_sampleLoss = new DTYPE*[timesize];
        m_numAnchor = (batchSize / 4);

        for (int i = 0; i < timesize; i++)
        {
            m_sampleLoss[i] = new DTYPE[m_numAnchor];
        }

        this->SetResult(new Tensor<DTYPE>(timesize, m_numAnchor, 1, 1, 1));

        return TRUE;
    }

    void Delete()
    {
        if (m_sampleLoss)
        {
            delete m_sampleLoss;
            m_sampleLoss = NULL;
        }
    }

    // float setAlpha(float alpha) { this->m_alpha = alpha; }
    // float setBeta(float beta) { this->m_beta = beta; }
    /*!
    @brief Quadruplet Loss의 순전파를 수행하는 메소드
    @param pTime 입력 Tensor의 Time축 Dimension
    @return LossFunction의 입력 Operator에 대한 Quadruplet Loss
    */
    Tensor<DTYPE>* ForwardPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* input = this->GetTensor();
        Tensor<DTYPE>* result = this->GetResult();
        Tensor<DTYPE>* alpha = this->m_alpha->GetResult();
        Tensor<DTYPE>* beta = this->m_beta->GetResult(); 

        int batchsize = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize = input->GetRowSize();
        int colsize = input->GetColSize();
        int featureDim = channelsize * rowsize * colsize;

        int time = pTime;
        int start = 0;
        int end = 0;

        for (int ba = 0; ba < m_numAnchor; ba++) {
            float dis_pos = 0.f;
            float dis_neg = 0.f;
            float dis_other_neg = 0.f;

            float d1 = 0.f, d2 = 0.f, d3 = 0.f;
            float loss = 0.f;

            int anc_idx = (pTime * batchsize + ba * 4) * featureDim;
            int anc_limit = anc_idx + featureDim;
            int pos_idx = anc_idx + featureDim;
            int neg_idx = pos_idx + featureDim;
            int other_neg_idx = neg_idx + featureDim;

            for (; anc_idx < anc_limit; anc_idx++, pos_idx++, neg_idx++, other_neg_idx++) 
            {
                // d1
                d1 = (*input)[anc_idx] - (*input)[pos_idx];
                dis_pos += d1 * d1;

                // d2
                d2 = (*input)[anc_idx] - (*input)[neg_idx];
                dis_neg += d2 * d2;

                // d3
                d3 = (*input)[neg_idx] - (*input)[other_neg_idx];
                dis_other_neg += d3 * d3;
            }

            float lossX = (dis_pos - dis_neg) + (*alpha)[ba];
            float lossY = (dis_pos - dis_other_neg) + (*beta)[ba];

            lossX /= featureDim;
            lossY /= featureDim;

            if (lossX < 0.f)
                lossX = 0.f;

            if (lossY < 0.f)
                lossY = 0.f;

            loss = lossX + lossY;
            // if (ba%2==0)
                // std::cout << "pos/neg/neg2/lossX/lossY/m1/m2 (loss): " << dis_pos << "/" << dis_neg << "/" << dis_other_neg << "/" << lossX << "/" << lossY << "/" << (*alpha)[ba] << "/" << (*beta)[ba]<< "(" << loss << ")" << std::endl;

            if (loss < 0.f)
                loss = 0.f;

            if (loss > 0.f)
            {
                (*result)[time * m_numAnchor + ba] = loss;
                m_sampleLoss[time][ba] = 1;
            }
            else
            {
                (*result)[time * m_numAnchor + ba] = 0;
                m_sampleLoss[time][ba] = 0;
            }
        }

        return result;
    }

    /*!
    @brief Quadruplet Loss의 역전파를 수행하는 메소드
    @details 구성한 뉴럴 네트워크에서 얻어진 Quadruplet LossFunction에 대한 Tensor의 Gradient를
    계산한다
    @param pTime 입력 Tensor의 Time 축의 Dimension
    @return NULL
    */
    Tensor<DTYPE>* BackPropagate(int pTime = 0)
    {
        Tensor<DTYPE>& input = *this->GetTensor();
        Tensor<DTYPE>& input_gradient = *this->GetOperator()->GetGradient();
        Tensor<DTYPE>* result = this->GetResult();

        int batchsize = input_gradient.GetBatchSize();
        int channelsize = input_gradient.GetChannelSize();
        int rowsize = input_gradient.GetRowSize();
        int colsize = input_gradient.GetColSize();

        int featureDim = channelsize * rowsize * colsize;

        for (int ba = 0; ba < m_numAnchor; ba++) {
            int anc_idx = (pTime * batchsize + ba * 4) * featureDim;
            int anc_limit = anc_idx + featureDim;

            int pos_idx = anc_idx + featureDim;
            int neg_idx = pos_idx + featureDim;
            int other_neg_idx = neg_idx + featureDim;

            if (m_sampleLoss[pTime][ba]) {
                for (; anc_idx < anc_limit; anc_idx++, pos_idx++, neg_idx++, other_neg_idx++)  {
                    input_gradient[anc_idx] = ((2.f * (input[anc_idx] + input[neg_idx])) - (4.f * input[pos_idx])) / featureDim;
                    input_gradient[pos_idx] = (4.f * (input[pos_idx] - input[anc_idx])) / featureDim;
                    input_gradient[neg_idx] = (2.f * (input[anc_idx] = 2.f * input[neg_idx] + input[other_neg_idx]))) / featureDim;
                    input_gradient[other_neg_idx] = (2.f * (input[neg_idx] - input[other_neg_idx])) / featureDim;
                }
            }
            else {
                memset(&input_gradient[anc_idx], 0, featureDim * sizeof(DTYPE) * 4);
            }
        }

        return NULL;
    }

#ifdef __CUDNN__

    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0)
    {
        this->ForwardPropagate();
        return NULL;
    }

    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0)
    {
        this->BackPropagate();
        return NULL;
    }

#endif // __CUDNN__
};

#endif // QUADRUPLETLOSS_H_
