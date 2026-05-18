#include "cudnn_mha_fwd_api.h"

// flash attention
#include "flash.h"
#include "static_switch.h"

// opague structs
struct cudnnAttnStruct
{
    unsigned attnMode;
    int nHeads;
    double smScaler;
    cudnnDataType_t dataType;
    cudnnDataType_t computePrec;
    cudnnMathType_t mathType;
    cudnnDropoutDescriptor_t attnDropoutDesc;
    cudnnDropoutDescriptor_t postDropoutDesc;
    int qSize;
    int kSize;
    int vSize;
    int qProjSize;
    int kProjSize;
    int vProjSize;
    int oProjSize;
    int qoMaxSeqLength;
    int kvMaxSeqLength;
    int maxBatchSize;
    int maxBeamSize;
};

cudnnStatus_t
cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc)
{
    if (attnDesc == NULL) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    struct cudnnAttnStruct *ptr = (struct cudnnAttnStruct *)malloc(sizeof(struct cudnnAttnStruct));
    
    if (ptr == NULL) {
        return CUDNN_STATUS_ALLOC_FAILED;
    }

    // Initialize all fields to 0 / NULL. 
    memset(ptr, 0, sizeof(struct cudnnAttnStruct));
    ptr->smScaler = 1.0; 
    *attnDesc = ptr;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t
cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc){
    if (attnDesc == NULL) {
        return CUDNN_STATUS_SUCCESS;
    }
    free(attnDesc);
    return CUDNN_STATUS_SUCCESS;
}

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
                       int maxBeamSize)
{
    if (attnDesc == NULL) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (nHeads <= 0 || maxBatchSize <= 0 || maxBeamSize <= 0) {
        return CUDNN_STATUS_BAD_PARAM;
    }

    if (qSize <= 0 || kSize <= 0 || vSize <= 0) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (qProjSize < 0 || kProjSize < 0 || vProjSize < 0 || oProjSize < 0) {
        return CUDNN_STATUS_BAD_PARAM;
    }

    if (qoMaxSeqLength <= 0 || kvMaxSeqLength <= 0) {
        return CUDNN_STATUS_BAD_PARAM;
    }

    // Check attnMode logic
    unsigned mapMask = CUDNN_ATTN_QUERYMAP_ONE_TO_ONE; // (1U << 0)
    unsigned biasMask = CUDNN_ATTN_ENABLE_PROJ_BIASES;  // (1U << 1)
    if (attnMode & ~(mapMask | biasMask)) {
        return CUDNN_STATUS_BAD_PARAM; 
    }


    attnDesc->attnMode        = attnMode;
    attnDesc->nHeads          = nHeads;
    attnDesc->smScaler        = smScaler;
    attnDesc->dataType        = dataType;
    attnDesc->computePrec     = computePrec;
    attnDesc->mathType        = mathType;
    attnDesc->attnDropoutDesc = attnDropoutDesc;
    attnDesc->postDropoutDesc = postDropoutDesc;
    attnDesc->qSize           = qSize;
    attnDesc->kSize           = kSize;
    attnDesc->vSize           = vSize;
    attnDesc->qProjSize       = qProjSize;
    attnDesc->kProjSize       = kProjSize;
    attnDesc->vProjSize       = vProjSize;
    attnDesc->oProjSize       = oProjSize;
    attnDesc->qoMaxSeqLength  = qoMaxSeqLength;
    attnDesc->kvMaxSeqLength  = kvMaxSeqLength;
    attnDesc->maxBatchSize    = maxBatchSize;
    attnDesc->maxBeamSize     = maxBeamSize;

    return CUDNN_STATUS_SUCCESS;
}

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
                       int *maxBeamSize)
{
    if (attnDesc == NULL) {
        return CUDNN_STATUS_BAD_PARAM;
    }

    if (attnMode)          *attnMode          = attnDesc->attnMode;
    if (nHeads)            *nHeads            = attnDesc->nHeads;
    if (smScaler)          *smScaler          = attnDesc->smScaler;
    if (dataType)          *dataType          = attnDesc->dataType;
    if (computePrec)       *computePrec       = attnDesc->computePrec;
    if (mathType)          *mathType          = attnDesc->mathType;
    if (attnDropoutDesc)   *attnDropoutDesc   = attnDesc->attnDropoutDesc;
    if (postDropoutDesc)   *postDropoutDesc   = attnDesc->postDropoutDesc;
    if (qSize)             *qSize             = attnDesc->qSize;
    if (kSize)             *kSize             = attnDesc->kSize;
    if (vSize)             *vSize             = attnDesc->vSize;
    if (qProjSize)         *qProjSize         = attnDesc->qProjSize;
    if (kProjSize)         *kProjSize         = attnDesc->kProjSize;
    if (vProjSize)         *vProjSize         = attnDesc->vProjSize;
    if (oProjSize)         *oProjSize         = attnDesc->oProjSize;
    if (qoMaxSeqLength)    *qoMaxSeqLength    = attnDesc->qoMaxSeqLength;
    if (kvMaxSeqLength)    *kvMaxSeqLength    = attnDesc->kvMaxSeqLength;
    if (maxBatchSize)      *maxBatchSize      = attnDesc->maxBatchSize;
    if (maxBeamSize)       *maxBeamSize       = attnDesc->maxBeamSize;

    return CUDNN_STATUS_SUCCESS;
}


cudnnStatus_t
cudnnMultiHeadAttnForward(cudnnHandle_t handle,
                          const cudnnAttnDescriptor_t attnDesc,
                          int currIdx,
                          const int loWinIdx[],
                          const int hiWinIdx[],
                          const int devSeqLengthsQO[],
                          const int devSeqLengthsKV[],
                          const cudnnSeqDataDescriptor_t qDesc,
                          const void *queries,
                          const void *residuals,
                          const cudnnSeqDataDescriptor_t kDesc,
                          const void *keys,
                          const cudnnSeqDataDescriptor_t vDesc,
                          const void *values,
                          const cudnnSeqDataDescriptor_t oDesc,
                          void *out,
                          size_t weightSizeInBytes,
                          const void *weights,
                          size_t workSpaceSizeInBytes,
                          void *workSpace,
                          size_t reserveSpaceSizeInBytes,
                          void *reserveSpace){










                            
                          }