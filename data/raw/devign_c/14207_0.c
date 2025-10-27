static inline void h264_idct8_1d(int16_t *block)

{

    __asm__ volatile(

        "movq 112(%0), %%mm7  \n\t"

        "movq  80(%0), %%mm0  \n\t"

        "movq  48(%0), %%mm3  \n\t"

        "movq  16(%0), %%mm5  \n\t"



        "movq   %%mm0, %%mm4  \n\t"

        "movq   %%mm5, %%mm1  \n\t"

        "psraw  $1,    %%mm4  \n\t"

        "psraw  $1,    %%mm1  \n\t"

        "paddw  %%mm0, %%mm4  \n\t"

        "paddw  %%mm5, %%mm1  \n\t"

        "paddw  %%mm7, %%mm4  \n\t"

        "paddw  %%mm0, %%mm1  \n\t"

        "psubw  %%mm5, %%mm4  \n\t"

        "paddw  %%mm3, %%mm1  \n\t"



        "psubw  %%mm3, %%mm5  \n\t"

        "psubw  %%mm3, %%mm0  \n\t"

        "paddw  %%mm7, %%mm5  \n\t"

        "psubw  %%mm7, %%mm0  \n\t"

        "psraw  $1,    %%mm3  \n\t"

        "psraw  $1,    %%mm7  \n\t"

        "psubw  %%mm3, %%mm5  \n\t"

        "psubw  %%mm7, %%mm0  \n\t"



        "movq   %%mm4, %%mm3  \n\t"

        "movq   %%mm1, %%mm7  \n\t"

        "psraw  $2,    %%mm1  \n\t"

        "psraw  $2,    %%mm3  \n\t"

        "paddw  %%mm5, %%mm3  \n\t"

        "psraw  $2,    %%mm5  \n\t"

        "paddw  %%mm0, %%mm1  \n\t"

        "psraw  $2,    %%mm0  \n\t"

        "psubw  %%mm4, %%mm5  \n\t"

        "psubw  %%mm0, %%mm7  \n\t"



        "movq  32(%0), %%mm2  \n\t"

        "movq  96(%0), %%mm6  \n\t"

        "movq   %%mm2, %%mm4  \n\t"

        "movq   %%mm6, %%mm0  \n\t"

        "psraw  $1,    %%mm4  \n\t"

        "psraw  $1,    %%mm6  \n\t"

        "psubw  %%mm0, %%mm4  \n\t"

        "paddw  %%mm2, %%mm6  \n\t"



        "movq    (%0), %%mm2  \n\t"

        "movq  64(%0), %%mm0  \n\t"

        SUMSUB_BA( %%mm0, %%mm2 )

        SUMSUB_BA( %%mm6, %%mm0 )

        SUMSUB_BA( %%mm4, %%mm2 )

        SUMSUB_BA( %%mm7, %%mm6 )

        SUMSUB_BA( %%mm5, %%mm4 )

        SUMSUB_BA( %%mm3, %%mm2 )

        SUMSUB_BA( %%mm1, %%mm0 )

        :: "r"(block)

    );

}
