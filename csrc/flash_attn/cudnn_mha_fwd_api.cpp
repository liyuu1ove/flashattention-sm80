#include "cudnn_mha_fwd_api.h"




// flash attention
#include "flash.h"
#include "static_switch.h"


// opague structs
struct cudnnAttnStruct{
    
};


cudnnStatus_t
cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc);

cudnnStatus_t
cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc);

cudnnStatus_t
cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
                       unsigned attnMode,
                       int nHeads,
                       double smScaler,
                       cudnnDataType_t dataType,
                       cudnnDataType_t computePrec,
                       cudnnMathType_t mathType,
                       cudnnDropoutDescriptor_t attnDropoutDesc,
                       cudnnDropoutDescriptor_t postDropoutDesc,
                       int qSize,
                       int kSize,
                       int vSize,
                       int qProjSize,
                       int kProjSize,
                       int vProjSize,
                       int oProjSize,
                       int qoMaxSeqLength,
                       int kvMaxSeqLength,
                       int maxBatchSize,
                       int maxBeamSize);

cudnnStatus_t 
cudnnGetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
                       unsigned *attnMode,
                       int *nHeads,
                       double *smScaler,
                       cudnnDataType_t *dataType,
                       cudnnDataType_t *computePrec,
                       cudnnMathType_t *mathType,
                       cudnnDropoutDescriptor_t *attnDropoutDesc,
                       cudnnDropoutDescriptor_t *postDropoutDesc,
                       int *qSize,
                       int *kSize,
                       int *vSize,
                       int *qProjSize,
                       int *kProjSize,
                       int *vProjSize,
                       int *oProjSize,
                       int *qoMaxSeqLength,
                       int *kvMaxSeqLength,
                       int *maxBatchSize,
                       int *maxBeamSize);