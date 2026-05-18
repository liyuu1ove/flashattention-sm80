/* Sequence data descriptor */

typedef enum {
    CUDNN_SEQDATA_TIME_DIM  = 0, /* index in time */
    CUDNN_SEQDATA_BATCH_DIM = 1, /* index in batch */
    CUDNN_SEQDATA_BEAM_DIM  = 2, /* index in beam */
    CUDNN_SEQDATA_VECT_DIM  = 3  /* index in vector */
} cudnnSeqDataAxis_t;

struct cudnnSeqDataStruct;
typedef struct cudnnSeqDataStruct *cudnnSeqDataDescriptor_t;

#define CUDNN_SEQDATA_DIM_COUNT 4 /* dimension count */

cudnnStatus_t CUDNNWINAPI
cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc);

cudnnStatus_t CUDNNWINAPI
cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc,
                          cudnnDataType_t dataType,
                          int nbDims,
                          const int dimA[],
                          const cudnnSeqDataAxis_t axes[],
                          size_t seqLengthArraySize,
                          const int seqLengthArray[],
                          void *paddingFill);

cudnnStatus_t CUDNNWINAPI
cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,
                          cudnnDataType_t *dataType,
                          int *nbDims,
                          int nbDimsRequested,
                          int dimA[],
                          cudnnSeqDataAxis_t axes[],
                          size_t *seqLengthArraySize,
                          size_t seqLengthSizeRequested,
                          int seqLengthArray[],
                          void *paddingFill);