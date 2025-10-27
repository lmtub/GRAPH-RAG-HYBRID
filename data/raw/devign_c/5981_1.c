static inline void RENAME(yuv2rgb555_2)(SwsContext *c, const uint16_t *buf0,

                                        const uint16_t *buf1, const uint16_t *ubuf0,

                                        const uint16_t *ubuf1, const uint16_t *vbuf0,

                                        const uint16_t *vbuf1, const uint16_t *abuf0,

                                        const uint16_t *abuf1, uint8_t *dest,

                                        int dstW, int yalpha, int uvalpha, int y)

{

    x86_reg uv_off = c->uv_off << 1;



    //Note 8280 == DSTW_OFFSET but the preprocessor can't handle that there :(

    __asm__ volatile(

        "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

        "mov        %4, %%"REG_b"               \n\t"

        "push %%"REG_BP"                        \n\t"

        YSCALEYUV2RGB(%%REGBP, %5, %6)

        "pxor    %%mm7, %%mm7                   \n\t"

        /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

        "paddusb "BLUE_DITHER"(%5), %%mm2      \n\t"

        "paddusb "GREEN_DITHER"(%5), %%mm4      \n\t"

        "paddusb "RED_DITHER"(%5), %%mm5      \n\t"

#endif

        WRITERGB15(%%REGb, 8280(%5), %%REGBP)

        "pop %%"REG_BP"                         \n\t"

        "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"

        :: "c" (buf0), "d" (buf1), "S" (ubuf0), "D" (ubuf1), "m" (dest),

           "a" (&c->redDither), "m"(uv_off)

    );

}
