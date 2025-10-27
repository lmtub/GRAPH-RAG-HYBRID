static inline void RENAME(yuv2packedX)(SwsContext *c, const int16_t *lumFilter, const int16_t **lumSrc, int lumFilterSize,

                                       const int16_t *chrFilter, const int16_t **chrSrc, int chrFilterSize,

                                       const int16_t **alpSrc, uint8_t *dest, long dstW, long dstY)

{

#if COMPILE_TEMPLATE_MMX

    x86_reg dummy=0;

    x86_reg dstW_reg = dstW;

    if(!(c->flags & SWS_BITEXACT)) {

        if (c->flags & SWS_ACCURATE_RND) {

            switch(c->dstFormat) {

            case PIX_FMT_RGB32:

                if (CONFIG_SWSCALE_ALPHA && c->alpPixBuf) {

                    YSCALEYUV2PACKEDX_ACCURATE

                    YSCALEYUV2RGBX

                    "movq                      %%mm2, "U_TEMP"(%0)  \n\t"

                    "movq                      %%mm4, "V_TEMP"(%0)  \n\t"

                    "movq                      %%mm5, "Y_TEMP"(%0)  \n\t"

                    YSCALEYUV2PACKEDX_ACCURATE_YA(ALP_MMX_FILTER_OFFSET)

                    "movq               "Y_TEMP"(%0), %%mm5         \n\t"

                    "psraw                        $3, %%mm1         \n\t"

                    "psraw                        $3, %%mm7         \n\t"

                    "packuswb                  %%mm7, %%mm1         \n\t"

                    WRITEBGR32(%4, %5, %%REGa, %%mm3, %%mm4, %%mm5, %%mm1, %%mm0, %%mm7, %%mm2, %%mm6)



                    YSCALEYUV2PACKEDX_END

                } else {

                    YSCALEYUV2PACKEDX_ACCURATE

                    YSCALEYUV2RGBX

                    "pcmpeqd %%mm7, %%mm7 \n\t"

                    WRITEBGR32(%4, %5, %%REGa, %%mm2, %%mm4, %%mm5, %%mm7, %%mm0, %%mm1, %%mm3, %%mm6)



                    YSCALEYUV2PACKEDX_END

                }

                return;

            case PIX_FMT_BGR24:

                YSCALEYUV2PACKEDX_ACCURATE

                YSCALEYUV2RGBX

                "pxor %%mm7, %%mm7 \n\t"

                "lea (%%"REG_a", %%"REG_a", 2), %%"REG_c"\n\t" //FIXME optimize

                "add %4, %%"REG_c"                        \n\t"

                WRITEBGR24(%%REGc, %5, %%REGa)





                :: "r" (&c->redDither),

                "m" (dummy), "m" (dummy), "m" (dummy),

                "r" (dest), "m" (dstW_reg)

                : "%"REG_a, "%"REG_c, "%"REG_d, "%"REG_S

                );

                return;

            case PIX_FMT_RGB555:

                YSCALEYUV2PACKEDX_ACCURATE

                YSCALEYUV2RGBX

                "pxor %%mm7, %%mm7 \n\t"

                /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

                "paddusb "BLUE_DITHER"(%0), %%mm2\n\t"

                "paddusb "GREEN_DITHER"(%0), %%mm4\n\t"

                "paddusb "RED_DITHER"(%0), %%mm5\n\t"

#endif



                WRITERGB15(%4, %5, %%REGa)

                YSCALEYUV2PACKEDX_END

                return;

            case PIX_FMT_RGB565:

                YSCALEYUV2PACKEDX_ACCURATE

                YSCALEYUV2RGBX

                "pxor %%mm7, %%mm7 \n\t"

                /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

                "paddusb "BLUE_DITHER"(%0), %%mm2\n\t"

                "paddusb "GREEN_DITHER"(%0), %%mm4\n\t"

                "paddusb "RED_DITHER"(%0), %%mm5\n\t"

#endif



                WRITERGB16(%4, %5, %%REGa)

                YSCALEYUV2PACKEDX_END

                return;

            case PIX_FMT_YUYV422:

                YSCALEYUV2PACKEDX_ACCURATE

                /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */



                "psraw $3, %%mm3    \n\t"

                "psraw $3, %%mm4    \n\t"

                "psraw $3, %%mm1    \n\t"

                "psraw $3, %%mm7    \n\t"

                WRITEYUY2(%4, %5, %%REGa)

                YSCALEYUV2PACKEDX_END

                return;

            }

        } else {

            switch(c->dstFormat) {

            case PIX_FMT_RGB32:

                if (CONFIG_SWSCALE_ALPHA && c->alpPixBuf) {

                    YSCALEYUV2PACKEDX

                    YSCALEYUV2RGBX

                    YSCALEYUV2PACKEDX_YA(ALP_MMX_FILTER_OFFSET, %%mm0, %%mm3, %%mm6, %%mm1, %%mm7)

                    "psraw                        $3, %%mm1         \n\t"

                    "psraw                        $3, %%mm7         \n\t"

                    "packuswb                  %%mm7, %%mm1         \n\t"

                    WRITEBGR32(%4, %5, %%REGa, %%mm2, %%mm4, %%mm5, %%mm1, %%mm0, %%mm7, %%mm3, %%mm6)

                    YSCALEYUV2PACKEDX_END

                } else {

                    YSCALEYUV2PACKEDX

                    YSCALEYUV2RGBX

                    "pcmpeqd %%mm7, %%mm7 \n\t"

                    WRITEBGR32(%4, %5, %%REGa, %%mm2, %%mm4, %%mm5, %%mm7, %%mm0, %%mm1, %%mm3, %%mm6)

                    YSCALEYUV2PACKEDX_END

                }

                return;

            case PIX_FMT_BGR24:

                YSCALEYUV2PACKEDX

                YSCALEYUV2RGBX

                "pxor                    %%mm7, %%mm7       \n\t"

                "lea (%%"REG_a", %%"REG_a", 2), %%"REG_c"   \n\t" //FIXME optimize

                "add                        %4, %%"REG_c"   \n\t"

                WRITEBGR24(%%REGc, %5, %%REGa)



                :: "r" (&c->redDither),

                "m" (dummy), "m" (dummy), "m" (dummy),

                "r" (dest),  "m" (dstW_reg)

                : "%"REG_a, "%"REG_c, "%"REG_d, "%"REG_S

                );

                return;

            case PIX_FMT_RGB555:

                YSCALEYUV2PACKEDX

                YSCALEYUV2RGBX

                "pxor %%mm7, %%mm7 \n\t"

                /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

                "paddusb "BLUE_DITHER"(%0), %%mm2  \n\t"

                "paddusb "GREEN_DITHER"(%0), %%mm4  \n\t"

                "paddusb "RED_DITHER"(%0), %%mm5  \n\t"

#endif



                WRITERGB15(%4, %5, %%REGa)

                YSCALEYUV2PACKEDX_END

                return;

            case PIX_FMT_RGB565:

                YSCALEYUV2PACKEDX

                YSCALEYUV2RGBX

                "pxor %%mm7, %%mm7 \n\t"

                /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

                "paddusb "BLUE_DITHER"(%0), %%mm2  \n\t"

                "paddusb "GREEN_DITHER"(%0), %%mm4  \n\t"

                "paddusb "RED_DITHER"(%0), %%mm5  \n\t"

#endif



                WRITERGB16(%4, %5, %%REGa)

                YSCALEYUV2PACKEDX_END

                return;

            case PIX_FMT_YUYV422:

                YSCALEYUV2PACKEDX

                /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */



                "psraw $3, %%mm3    \n\t"

                "psraw $3, %%mm4    \n\t"

                "psraw $3, %%mm1    \n\t"

                "psraw $3, %%mm7    \n\t"

                WRITEYUY2(%4, %5, %%REGa)

                YSCALEYUV2PACKEDX_END

                return;

            }

        }

    }

#endif /* COMPILE_TEMPLATE_MMX */

#if COMPILE_TEMPLATE_ALTIVEC

    /* The following list of supported dstFormat values should

       match what's found in the body of ff_yuv2packedX_altivec() */

    if (!(c->flags & SWS_BITEXACT) && !c->alpPixBuf &&

         (c->dstFormat==PIX_FMT_ABGR  || c->dstFormat==PIX_FMT_BGRA  ||

          c->dstFormat==PIX_FMT_BGR24 || c->dstFormat==PIX_FMT_RGB24 ||

          c->dstFormat==PIX_FMT_RGBA  || c->dstFormat==PIX_FMT_ARGB))

            ff_yuv2packedX_altivec(c, lumFilter, lumSrc, lumFilterSize,

                                   chrFilter, chrSrc, chrFilterSize,

                                   dest, dstW, dstY);

    else

#endif

        yuv2packedXinC(c, lumFilter, lumSrc, lumFilterSize,

                       chrFilter, chrSrc, chrFilterSize,

                       alpSrc, dest, dstW, dstY);

}
