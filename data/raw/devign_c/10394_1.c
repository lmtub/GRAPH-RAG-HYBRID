static inline void RENAME(bgr24ToY_mmx)(uint8_t *dst, const uint8_t *src, int width, enum PixelFormat srcFormat)

{



    if(srcFormat == PIX_FMT_BGR24) {

        __asm__ volatile(

            "movq  "MANGLE(ff_bgr24toY1Coeff)", %%mm5       \n\t"

            "movq  "MANGLE(ff_bgr24toY2Coeff)", %%mm6       \n\t"

            :

        );

    } else {

        __asm__ volatile(

            "movq  "MANGLE(ff_rgb24toY1Coeff)", %%mm5       \n\t"

            "movq  "MANGLE(ff_rgb24toY2Coeff)", %%mm6       \n\t"

            :

        );

    }



    __asm__ volatile(

        "movq  "MANGLE(ff_bgr24toYOffset)", %%mm4   \n\t"

        "mov                        %2, %%"REG_a"   \n\t"

        "pxor                    %%mm7, %%mm7       \n\t"

        "1:                                         \n\t"

        PREFETCH"               64(%0)              \n\t"

        "movd                     (%0), %%mm0       \n\t"

        "movd                    2(%0), %%mm1       \n\t"

        "movd                    6(%0), %%mm2       \n\t"

        "movd                    8(%0), %%mm3       \n\t"

        "add                       $12, %0          \n\t"

        "punpcklbw               %%mm7, %%mm0       \n\t"

        "punpcklbw               %%mm7, %%mm1       \n\t"

        "punpcklbw               %%mm7, %%mm2       \n\t"

        "punpcklbw               %%mm7, %%mm3       \n\t"

        "pmaddwd                 %%mm5, %%mm0       \n\t"

        "pmaddwd                 %%mm6, %%mm1       \n\t"

        "pmaddwd                 %%mm5, %%mm2       \n\t"

        "pmaddwd                 %%mm6, %%mm3       \n\t"

        "paddd                   %%mm1, %%mm0       \n\t"

        "paddd                   %%mm3, %%mm2       \n\t"

        "paddd                   %%mm4, %%mm0       \n\t"

        "paddd                   %%mm4, %%mm2       \n\t"

        "psrad                     $15, %%mm0       \n\t"

        "psrad                     $15, %%mm2       \n\t"

        "packssdw                %%mm2, %%mm0       \n\t"

        "packuswb                %%mm0, %%mm0       \n\t"

        "movd                %%mm0, (%1, %%"REG_a") \n\t"

        "add                        $4, %%"REG_a"   \n\t"

        " js                        1b              \n\t"

    : "+r" (src)

    : "r" (dst+width), "g" ((x86_reg)-width)

    : "%"REG_a

    );

}
