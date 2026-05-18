#include "shared_struct.h"
#include "seqdata_descriptor.h"

/*
Not implemented functions
1. beam search
2. 


*/ 



/* Multihead Attention */

struct cudnnAttnStruct;
typedef struct cudnnAttnStruct *cudnnAttnDescriptor_t;


/* Legacy type for backward compatibility */
typedef unsigned cudnnAttnQueryMap_t;

/*
 * Multi-head attention options passed via 'attnMode' in cudnnSetAttnDescriptor().
 * Use the bitwise OR operator to combine several settings listed below.  Additional
 * minor options can be added here w/o changing or introducing new API functions.
 */
#define CUDNN_ATTN_QUERYMAP_ALL_TO_ONE 0         /* multiple Q-s map to a single (K,V) set when beam size > 1 */
#define CUDNN_ATTN_QUERYMAP_ONE_TO_ONE (1U << 0) /* multiple Q-s map to multiple (K,V) sets when beam size > 1 */
#define CUDNN_ATTN_DISABLE_PROJ_BIASES 0         /* no biases in attention input and output projections */
#define CUDNN_ATTN_ENABLE_PROJ_BIASES (1U << 1)  /* use biases in attention input and output projections */

typedef enum {
    CUDNN_MH_ATTN_Q_WEIGHTS = 0, /* input projection weights for 'queries' */
    CUDNN_MH_ATTN_K_WEIGHTS = 1, /* input projection weights for 'keys' */
    CUDNN_MH_ATTN_V_WEIGHTS = 2, /* input projection weights for 'values' */
    CUDNN_MH_ATTN_O_WEIGHTS = 3, /* output projection weights */
    CUDNN_MH_ATTN_Q_BIASES  = 4, /* input projection bias tensor for 'queries' */
    CUDNN_MH_ATTN_K_BIASES  = 5, /* input projection bias for 'keys' */
    CUDNN_MH_ATTN_V_BIASES  = 6, /* input projection bias for 'values' */
    CUDNN_MH_ATTN_O_BIASES  = 7, /* output projection biases */
} cudnnMultiHeadAttnWeightKind_t;

#define CUDNN_ATTN_WKIND_COUNT 8 /* Number of attention weight/bias tensors */

extern "C" {
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

cudnnStatus_t 
cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                             const cudnnAttnDescriptor_t attnDesc,
                             size_t *weightSizeInBytes,
                             size_t *workSpaceSizeInBytes,
                             size_t *reserveSpaceSizeInBytes);



cudnnStatus_t 
cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle,
                             const cudnnAttnDescriptor_t attnDesc,
                             cudnnMultiHeadAttnWeightKind_t wKind,
                             size_t weightSizeInBytes,
                             const void *weights,
                             cudnnTensorDescriptor_t wDesc,
                             void **wAddr);

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
                          void *reserveSpace);

}