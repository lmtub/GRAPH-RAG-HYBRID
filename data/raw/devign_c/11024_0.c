static int packed_16bpc_bswap(SwsContext *c, const uint8_t *src[],

                              int srcStride[], int srcSliceY, int srcSliceH,

                              uint8_t *dst[], int dstStride[])

{

    int i, j, p;



    for (p = 0; p < 4; p++) {

        int srcstr = srcStride[p] >> 1;

        int dststr = dstStride[p] >> 1;

        uint16_t       *dstPtr =       (uint16_t *) dst[p];

        const uint16_t *srcPtr = (const uint16_t *) src[p];

        int min_stride         = FFMIN(srcstr, dststr);

        if(!dstPtr || !srcPtr)

            continue;

        for (i = 0; i < (srcSliceH >> c->chrDstVSubSample); i++) {

            for (j = 0; j < min_stride; j++) {

                dstPtr[j] = av_bswap16(srcPtr[j]);

            }

            srcPtr += srcstr;

            dstPtr += dststr;

        }

    }



    return srcSliceH;

}
