/* Sequence data descriptor */

typedef enum {
    XYDNN_SEQDATA_TIME_DIM  = 0, /* index in time */
    XYDNN_SEQDATA_BATCH_DIM = 1, /* index in batch */
    XYDNN_SEQDATA_BEAM_DIM  = 2, /* index in beam */
    XYDNN_SEQDATA_VECT_DIM  = 3  /* index in vector */
} xydnnSeqDataAxis_t;

struct xydnnSeqDataStruct;
typedef struct xydnnSeqDataStruct *xydnnSeqDataDescriptor_t;

#define XYDNN_SEQDATA_DIM_COUNT 4 /* dimension count */

extern "C" {

xydnnStatus_t XYDNNWINAPI
xydnnCreateSeqDataDescriptor(xydnnSeqDataDescriptor_t *seqDataDesc);

xydnnStatus_t XYDNNWINAPI
xydnnDestroySeqDataDescriptor(xydnnSeqDataDescriptor_t seqDataDesc);

xydnnStatus_t XYDNNWINAPI
xydnnSetSeqDataDescriptor(xydnnSeqDataDescriptor_t seqDataDesc,
                          xydnnDataType_t dataType,
                          int nbDims,
                          const int dimA[],
                          const xydnnSeqDataAxis_t axes[],
                          size_t seqLengthArraySize,
                          const int seqLengthArray[],
                          void *paddingFill);

xydnnStatus_t XYDNNWINAPI
xydnnGetSeqDataDescriptor(const xydnnSeqDataDescriptor_t seqDataDesc,
                          xydnnDataType_t *dataType,
                          int *nbDims,
                          int nbDimsRequested,
                          int dimA[],
                          xydnnSeqDataAxis_t axes[],
                          size_t *seqLengthArraySize,
                          size_t seqLengthSizeRequested,
                          int seqLengthArray[],
                          void *paddingFill);

}
