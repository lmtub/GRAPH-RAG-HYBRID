inline static void RENAME(hcscale)(SwsContext *c, uint16_t *dst, int dstWidth, const uint8_t *src1, const uint8_t *src2,

                                   int srcW, int xInc, const int16_t *hChrFilter,

                                   const int16_t *hChrFilterPos, int hChrFilterSize,

                                   uint8_t *formatConvBuffer,

                                   uint32_t *pal)

{



    src1 += c->chrSrcOffset;

    src2 += c->chrSrcOffset;



    if (c->chrToYV12) {

        c->chrToYV12(formatConvBuffer, formatConvBuffer+VOFW, src1, src2, srcW, pal);

        src1= formatConvBuffer;

        src2= formatConvBuffer+VOFW;

    }



    if (!c->hcscale_fast) {

        c->hScale(dst     , dstWidth, src1, srcW, xInc, hChrFilter, hChrFilterPos, hChrFilterSize);

        c->hScale(dst+VOFW, dstWidth, src2, srcW, xInc, hChrFilter, hChrFilterPos, hChrFilterSize);

    } else { // fast bilinear upscale / crap downscale

        c->hcscale_fast(c, dst, dstWidth, src1, src2, srcW, xInc);

    }



    if (c->chrConvertRange)

        c->chrConvertRange(dst, dstWidth);

}
