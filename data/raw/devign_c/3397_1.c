static av_always_inline void hcscale(SwsContext *c, int16_t *dst1, int16_t *dst2, int dstWidth,

                                     const uint8_t *src_in[4],

                                     int srcW, int xInc, const int16_t *hChrFilter,

                                     const int16_t *hChrFilterPos, int hChrFilterSize,

                                     uint8_t *formatConvBuffer, uint32_t *pal)

{

    const uint8_t *src1 = src_in[1], *src2 = src_in[2];

    if (c->chrToYV12) {

        uint8_t *buf2 = formatConvBuffer + FFALIGN(srcW * FFALIGN(c->srcBpc, 8) >> 3, 16);

        c->chrToYV12(formatConvBuffer, buf2, src1, src2, srcW, pal);

        src1= formatConvBuffer;

        src2= buf2;

    } else if (c->readChrPlanar) {

        uint8_t *buf2 = formatConvBuffer + FFALIGN(srcW * FFALIGN(c->srcBpc, 8) >> 3, 16);

        c->readChrPlanar(formatConvBuffer, buf2, src_in, srcW);

        src1= formatConvBuffer;

        src2= buf2;

    }



    if (!c->hcscale_fast) {

        c->hcScale(c, dst1, dstWidth, src1, hChrFilter, hChrFilterPos, hChrFilterSize);

        c->hcScale(c, dst2, dstWidth, src2, hChrFilter, hChrFilterPos, hChrFilterSize);

    } else { // fast bilinear upscale / crap downscale

        c->hcscale_fast(c, dst1, dst2, dstWidth, src1, src2, srcW, xInc);

    }



    if (c->chrConvertRange)

        c->chrConvertRange(dst1, dst2, dstWidth);

}
