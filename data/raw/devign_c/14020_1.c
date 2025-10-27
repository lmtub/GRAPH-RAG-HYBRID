static void postprocess_chroma(AVFrame *frame, int w, int h, int depth)

{

    uint16_t *dstu = (uint16_t *)frame->data[1];

    uint16_t *dstv = (uint16_t *)frame->data[2];

    int16_t *srcu  = (int16_t *)frame->data[1];

    int16_t *srcv  = (int16_t *)frame->data[2];

    ptrdiff_t strideu = frame->linesize[1] / 2;

    ptrdiff_t stridev = frame->linesize[2] / 2;

    const int add = 1 << (depth - 1);

    const int shift = 16 - depth;

    int i, j;



    for (j = 0; j < h; j++) {

        for (i = 0; i < w; i++) {

            dstu[i] = (add + srcu[i]) << shift;

            dstv[i] = (add + srcv[i]) << shift;

        }

        dstu += strideu;

        dstv += stridev;

        srcu += strideu;

        srcv += stridev;

    }

}
