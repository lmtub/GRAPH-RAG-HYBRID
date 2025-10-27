static inline void RENAME(hyscale)(SwsContext *c, uint16_t *dst, int dstWidth, const uint8_t *src, int srcW, int xInc,

                                   const int16_t *hLumFilter,

                                   const int16_t *hLumFilterPos, int hLumFilterSize,

                                   uint8_t *formatConvBuffer,

                                   uint32_t *pal, int isAlpha)

{

    void (*toYV12)(uint8_t *, const uint8_t *, int, uint32_t *) = isAlpha ? c->alpToYV12 : c->lumToYV12;

    void (*convertRange)(uint16_t *, int) = isAlpha ? NULL : c->lumConvertRange;



    src += isAlpha ? c->alpSrcOffset : c->lumSrcOffset;



    if (toYV12) {

        toYV12(formatConvBuffer, src, srcW, pal);

        src= formatConvBuffer;

    }



    if (!c->hyscale_fast) {

        c->hScale(dst, dstWidth, src, srcW, xInc, hLumFilter, hLumFilterPos, hLumFilterSize);

    } else { // fast bilinear upscale / crap downscale

        c->hyscale_fast(c, dst, dstWidth, src, srcW, xInc);

    }



    if (convertRange)

        convertRange(dst, dstWidth);

}
