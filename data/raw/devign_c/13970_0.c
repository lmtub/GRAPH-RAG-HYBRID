static inline void RENAME(bgr24ToUV_mmx)(uint8_t *dstU, uint8_t *dstV, const uint8_t *src, long width, enum PixelFormat srcFormat)

{

    __asm__ volatile(

        "movq                   24(%4), %%mm6       \n\t"

        "mov                        %3, %%"REG_a"   \n\t"

        "pxor                    %%mm7, %%mm7       \n\t"

        "1:                                         \n\t"

        PREFETCH"               64(%0)              \n\t"

        "movd                     (%0), %%mm0       \n\t"

        "movd                    2(%0), %%mm1       \n\t"

        "punpcklbw               %%mm7, %%mm0       \n\t"

        "punpcklbw               %%mm7, %%mm1       \n\t"

        "movq                    %%mm0, %%mm2       \n\t"

        "movq                    %%mm1, %%mm3       \n\t"

        "pmaddwd                  (%4), %%mm0       \n\t"

        "pmaddwd                 8(%4), %%mm1       \n\t"

        "pmaddwd                16(%4), %%mm2       \n\t"

        "pmaddwd                 %%mm6, %%mm3       \n\t"

        "paddd                   %%mm1, %%mm0       \n\t"

        "paddd                   %%mm3, %%mm2       \n\t"



        "movd                    6(%0), %%mm1       \n\t"

        "movd                    8(%0), %%mm3       \n\t"

        "add                       $12, %0          \n\t"

        "punpcklbw               %%mm7, %%mm1       \n\t"

        "punpcklbw               %%mm7, %%mm3       \n\t"

        "movq                    %%mm1, %%mm4       \n\t"

        "movq                    %%mm3, %%mm5       \n\t"

        "pmaddwd                  (%4), %%mm1       \n\t"

        "pmaddwd                 8(%4), %%mm3       \n\t"

        "pmaddwd                16(%4), %%mm4       \n\t"

        "pmaddwd                 %%mm6, %%mm5       \n\t"

        "paddd                   %%mm3, %%mm1       \n\t"

        "paddd                   %%mm5, %%mm4       \n\t"



        "movq "MANGLE(ff_bgr24toUVOffset)", %%mm3       \n\t"

        "paddd                   %%mm3, %%mm0       \n\t"

        "paddd                   %%mm3, %%mm2       \n\t"

        "paddd                   %%mm3, %%mm1       \n\t"

        "paddd                   %%mm3, %%mm4       \n\t"

        "psrad                     $15, %%mm0       \n\t"

        "psrad                     $15, %%mm2       \n\t"

        "psrad                     $15, %%mm1       \n\t"

        "psrad                     $15, %%mm4       \n\t"

        "packssdw                %%mm1, %%mm0       \n\t"

        "packssdw                %%mm4, %%mm2       \n\t"

        "packuswb                %%mm0, %%mm0       \n\t"

        "packuswb                %%mm2, %%mm2       \n\t"

        "movd                %%mm0, (%1, %%"REG_a") \n\t"

        "movd                %%mm2, (%2, %%"REG_a") \n\t"

        "add                        $4, %%"REG_a"   \n\t"

        " js                        1b              \n\t"

    : "+r" (src)

    : "r" (dstU+width), "r" (dstV+width), "g" ((x86_reg)-width), "r"(ff_bgr24toUV[srcFormat == PIX_FMT_RGB24])

    : "%"REG_a

    );

}
