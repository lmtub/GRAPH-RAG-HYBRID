static inline void RENAME(yuv2yuv1)(SwsContext *c, const int16_t *lumSrc, const int16_t *chrSrc, const int16_t *alpSrc,

                                    uint8_t *dest, uint8_t *uDest, uint8_t *vDest, uint8_t *aDest, int dstW, int chrDstW)

{

    int i;

#if COMPILE_TEMPLATE_MMX

    if(!(c->flags & SWS_BITEXACT)) {

        long p= 4;

        const uint8_t *src[4]= {alpSrc + dstW, lumSrc + dstW, chrSrc + chrDstW, chrSrc + VOFW + chrDstW};

        uint8_t *dst[4]= {aDest, dest, uDest, vDest};

        x86_reg counter[4]= {dstW, dstW, chrDstW, chrDstW};



        if (c->flags & SWS_ACCURATE_RND) {

            while(p--) {

                if (dst[p]) {

                    __asm__ volatile(

                        YSCALEYUV2YV121_ACCURATE

                        :: "r" (src[p]), "r" (dst[p] + counter[p]),

                        "g" (-counter[p])

                        : "%"REG_a

                    );

                }

            }

        } else {

            while(p--) {

                if (dst[p]) {

                    __asm__ volatile(

                        YSCALEYUV2YV121

                        :: "r" (src[p]), "r" (dst[p] + counter[p]),

                        "g" (-counter[p])

                        : "%"REG_a

                    );

                }

            }

        }

        return;

    }

#endif

    for (i=0; i<dstW; i++) {

        int val= (lumSrc[i]+64)>>7;



        if (val&256) {

            if (val<0) val=0;

            else       val=255;

        }



        dest[i]= val;

    }



    if (uDest)

        for (i=0; i<chrDstW; i++) {

            int u=(chrSrc[i       ]+64)>>7;

            int v=(chrSrc[i + VOFW]+64)>>7;



            if ((u|v)&256) {

                if (u<0)        u=0;

                else if (u>255) u=255;

                if (v<0)        v=0;

                else if (v>255) v=255;

            }



            uDest[i]= u;

            vDest[i]= v;

        }



    if (CONFIG_SWSCALE_ALPHA && aDest)

        for (i=0; i<dstW; i++) {

            int val= (alpSrc[i]+64)>>7;

            aDest[i]= av_clip_uint8(val);

        }

}
