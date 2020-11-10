Passer.h


operator.hpp > container.hpp > common.h > #include <stdarg.h> 선언되어 있음

```
int Alloc(int numOperator, ...)
```

가변 인자 함수, 최소 1개이상의 고정 인수 있어야 함, ...은 가장 마지막 파라미터에

va_list ap;	// 인자 리스트에서 추출하고 싶은 인자의 주소를 가리키는 포인터
va_start(ap, numOperator);	//ap가 맨 첫번째 가변인수를 가리키도록 초기화


va_list: 가변 인자 목록. 가변 인자의 메모리 주소를 저장하는 포인터
va_start: 가변 인자를 가져올 수 있도록 포인터를 설정
va_arg: 가변 인자 포인터에서 특정 자료형 크기만큼 값을 가져옴
va_end: 가변 인자 처리가 끝났을 때 포인터를 NULL로 초기화

```
#ifndef PASSER_H_
#define PASSER_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class Passer : public Operator<DTYPE>{
private:
    int m_noOperator;
    int *m_aAccumulate;

public:
    Passer(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, Operator<DTYPE> *pInput2, 
            Operator<DTYPE> *pInput3, std::string pName = "NO NAME", int pLoadflag = TRUE) 
            : Operator<DTYPE>(pInput0, pInput1, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Passer::Passer(Operator *)" << '\n';
        #endif  // __DEBUG__

        m_noOperator = 0;
        this->Alloc(4, pInput0, pInput1, pInput2, pInput3);
    }

    ~Passer() {
        #ifdef __DEBUG__
        std::cout << "Passer::~Passer()" << '\n';
        #endif  // __DEBUG__

        Delete()
    }

    int Alloc(int noOperator, ...) {
        #ifdef __DEBUG__
        std::cout << "Passer::Alloc(Operator *, Operator *, Operator *, Operator *,)" << '\n';
        #endif  // __DEBUG__

        m_noOperator  = noOperator;
        m_aAccumulate = new int[noOperator];

        // 파라미터의 ...에 해당하는 인자들을 ap로 접근
        va_list ap;
        // 가변인수 접근 (가변인자 포인터, 가변인자 갯수)
        va_start(ap, noOperator);

        // 실제로 값을 가지고 오는 부분. (가변인자 포인터, 타입)
        Operator<DTYPE> *temp = va_arg(ap, Operator<DTYPE> *);
        
        int timesize    = temp->GetResult()->GetTimeSize();
        int batchsize   = temp->GetResult()->GetBatchSize();
        int channelsize = temp->GetResult()->GetChannelSize();
        int rowsize     = temp->GetResult()->GetRowSize();
        int colsize     = temp->GetResult()->GetColSize();

        int totalchannelsize = channelsize;

        m_aAccumulate[0] = 0;
        m_aAccumulate[1] = channelsize;
        m_aAccumulate[2] = 
        m_aAccumulate[3] =

        for (int i = 1; i < noOperator; i++) {
            temp = va_arg(ap, Operator<DTYPE> *);

            totalchannelsize += temp->GetResult()->GetChannelSize();

            if (i != noOperator - 1) m_aAccumulate[i + 1] = totalchannelsize;
    
        }

        va_end(ap);

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, totalchannelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, totalchannelsize, rowsize, colsize));

        return TRUE;
    }

    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = NULL;
        Tensor<DTYPE> *result = this->GetResult();

        int timesize  = result->GetTimeSize();
        int batchsize = result->GetBatchSize();
        int rowsize   = result->GetRowSize();
        int colsize   = result->GetColSize();

        Shape *inputTenShape  = NULL;
        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        // int totalchannelsize = 0;

        for (int i = 0; i < m_noOperator; i++) {
            input         = this->GetInput()[i]->GetResult();
            inputTenShape = input->GetShape();
            int channelsize = input->GetChannelSize();

            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            (*result)[Index5D(resultTenShape, ti, ba, m_aAccumulate[i] + ch, ro, co)]
                                = (*input)[Index5D(inputTenShape, ti, ba, ch, ro, co)];
                        }
                    }
                }
            }

            // totalchannelsize += channelsize;
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = NULL;

        int timesize  = this_delta->GetTimeSize();
        int batchsize = this_delta->GetBatchSize();
        int rowsize   = this_delta->GetRowSize();
        int colsize   = this_delta->GetColSize();

        Shape *inputTenShape  = NULL;
        Shape *resultTenShape = this_delta->GetShape();

        int ti = pTime;

        for (int i = 0; i < m_noOperator; i++) {
            input_delta   = this->GetInput()[i]->GetDelta();
            inputTenShape = input_delta->GetShape();
            int channelsize = input_delta->GetChannelSize();

            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            (*input_delta)[Index5D(inputTenShape, ti, ba, ch, ro, co)]
                                += (*this_delta)[Index5D(resultTenShape, ti, ba, m_aAccumulate[i] + ch, ro, co)];
                        }
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);

    int BackPropagateOnGPU(int pTime);

#endif  // __CUDNN__
};

#endif  // CONCATENATECHANNELWISE_H_
```


