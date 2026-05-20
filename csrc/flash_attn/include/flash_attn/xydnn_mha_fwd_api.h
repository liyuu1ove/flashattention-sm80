#include "shared_struct.h"
#include "seqdata_descriptor.h"

/*
Not implemented functions
1. beam search
2. 


*/ 



/* Multihead Attention */

struct xydnnAttnStruct;
typedef struct xydnnAttnStruct *xydnnAttnDescriptor_t;


/* Legacy type for backward compatibility */
typedef unsigned xydnnAttnQueryMap_t;

/*
 * Multi-head attention options passed via 'attnMode' in xydnnSetAttnDescriptor().
 * Use the bitwise OR operator to combine several settings listed below.  Additional
 * minor options can be added here w/o changing or introducing new API functions.
 */
#define XYDNN_ATTN_QUERYMAP_ALL_TO_ONE 0         /* multiple Q-s map to a single (K,V) set when beam size > 1 */
#define XYDNN_ATTN_QUERYMAP_ONE_TO_ONE (1U << 0) /* multiple Q-s map to multiple (K,V) sets when beam size > 1 */
#define XYDNN_ATTN_DISABLE_PROJ_BIASES 0         /* no biases in attention input and output projections */
#define XYDNN_ATTN_ENABLE_PROJ_BIASES (1U << 1)  /* use biases in attention input and output projections */

typedef enum {
    XYDNN_MH_ATTN_Q_WEIGHTS = 0, /* input projection weights for 'queries' */
    XYDNN_MH_ATTN_K_WEIGHTS = 1, /* input projection weights for 'keys' */
    XYDNN_MH_ATTN_V_WEIGHTS = 2, /* input projection weights for 'values' */
    XYDNN_MH_ATTN_O_WEIGHTS = 3, /* output projection weights */
    XYDNN_MH_ATTN_Q_BIASES  = 4, /* input projection bias tensor for 'queries' */
    XYDNN_MH_ATTN_K_BIASES  = 5, /* input projection bias for 'keys' */
    XYDNN_MH_ATTN_V_BIASES  = 6, /* input projection bias for 'values' */
    XYDNN_MH_ATTN_O_BIASES  = 7, /* output projection biases */
} xydnnMultiHeadAttnWeightKind_t;

#define XYDNN_ATTN_WKIND_COUNT 8 /* Number of attention weight/bias tensors */

extern "C" {

xydnnStatus_t XYDNNWINAPI
xydnnCreateTensorDescriptor(xydnnTensorDescriptor_t *tensorDesc);

xydnnStatus_t XYDNNWINAPI
xydnnDestroyTensorDescriptor(xydnnTensorDescriptor_t tensorDesc);

xydnnStatus_t XYDNNWINAPI
xydnnSetTensorNdDescriptor(xydnnTensorDescriptor_t tensorDesc,
                           xydnnDataType_t dataType,
                           int nbDims,
                           const int dimA[],
                           const int strideA[]);

xydnnStatus_t XYDNNWINAPI
xydnnGetTensorNdDescriptor(const xydnnTensorDescriptor_t tensorDesc,
                           int nbDimsRequested,
                           xydnnDataType_t *dataType,
                           int *nbDims,
                           int dimA[],
                           int strideA[]);

xydnnStatus_t
xydnnCreateAttnDescriptor(xydnnAttnDescriptor_t *attnDesc);

xydnnStatus_t
xydnnDestroyAttnDescriptor(xydnnAttnDescriptor_t attnDesc);

xydnnStatus_t
xydnnSetAttnDescriptor(xydnnAttnDescriptor_t attnDesc,
                       unsigned attnMode,
                       int nHeads,
                       double smScaler,
                       xydnnDataType_t dataType,
                       xydnnDataType_t computePrec,
                       xydnnMathType_t mathType,
                       xydnnDropoutDescriptor_t attnDropoutDesc,
                       xydnnDropoutDescriptor_t postDropoutDesc,
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

xydnnStatus_t 
xydnnGetAttnDescriptor(xydnnAttnDescriptor_t attnDesc,
                       unsigned *attnMode,
                       int *nHeads,
                       double *smScaler,
                       xydnnDataType_t *dataType,
                       xydnnDataType_t *computePrec,
                       xydnnMathType_t *mathType,
                       xydnnDropoutDescriptor_t *attnDropoutDesc,
                       xydnnDropoutDescriptor_t *postDropoutDesc,
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

xydnnStatus_t
xydnnSetAttnDescriptorOptions(xydnnAttnDescriptor_t attnDesc,
                              int isCausal,
                              int windowSizeLeft,
                              int windowSizeRight,
                              float softcap);

xydnnStatus_t
xydnnGetAttnDescriptorOptions(xydnnAttnDescriptor_t attnDesc,
                              int *isCausal,
                              int *windowSizeLeft,
                              int *windowSizeRight,
                              float *softcap);

xydnnStatus_t
xydnnSetAttnDescriptorAlibiSlopes(xydnnAttnDescriptor_t attnDesc,
                                  const void *alibiSlopes,
                                  int batchStride);

xydnnStatus_t
xydnnGetAttnDescriptorAlibiSlopes(xydnnAttnDescriptor_t attnDesc,
                                  const void **alibiSlopes,
                                  int *batchStride);

xydnnStatus_t
xydnnSetAttnDescriptorVarlen(xydnnAttnDescriptor_t attnDesc,
                             const int *cuSeqlensQO,
                             const int *cuSeqlensKV,
                             int totalQO,
                             int totalKV);

xydnnStatus_t
xydnnGetAttnDescriptorVarlen(xydnnAttnDescriptor_t attnDesc,
                             const int **cuSeqlensQO,
                             const int **cuSeqlensKV,
                             int *totalQO,
                             int *totalKV);

xydnnStatus_t 
xydnnGetMultiHeadAttnBuffers(xydnnHandle_t handle,
                             const xydnnAttnDescriptor_t attnDesc,
                             size_t *weightSizeInBytes,
                             size_t *workSpaceSizeInBytes,
                             size_t *reserveSpaceSizeInBytes);



xydnnStatus_t 
xydnnGetMultiHeadAttnWeights(xydnnHandle_t handle,
                             const xydnnAttnDescriptor_t attnDesc,
                             xydnnMultiHeadAttnWeightKind_t wKind,
                             size_t weightSizeInBytes,
                             const void *weights,
                             xydnnTensorDescriptor_t wDesc,
                             void **wAddr);

xydnnStatus_t
xydnnMultiHeadAttnForward(xydnnHandle_t handle,
                          const xydnnAttnDescriptor_t attnDesc,
                          int currIdx,
                          const int loWinIdx[],
                          const int hiWinIdx[],
                          const int devSeqLengthsQO[],
                          const int devSeqLengthsKV[],
                          const xydnnSeqDataDescriptor_t qDesc,
                          const void *queries,
                          const void *residuals,
                          const xydnnSeqDataDescriptor_t kDesc,
                          const void *keys,
                          const xydnnSeqDataDescriptor_t vDesc,
                          const void *values,
                          const xydnnSeqDataDescriptor_t oDesc,
                          void *out,
                          size_t weightSizeInBytes,
                          const void *weights,
                          size_t workSpaceSizeInBytes,
                          void *workSpace,
                          size_t reserveSpaceSizeInBytes,
                          void *reserveSpace);

}
