static inline void RENAME(planar2x)(const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight, int srcStride, int dstStride)

{

    int x,y;



    dst[0]= src[0];



    // first line

    for (x=0; x<srcWidth-1; x++) {

        dst[2*x+1]= (3*src[x] +   src[x+1])>>2;

        dst[2*x+2]= (  src[x] + 3*src[x+1])>>2;

    }

    dst[2*srcWidth-1]= src[srcWidth-1];



    dst+= dstStride;



    for (y=1; y<srcHeight; y++) {

        const x86_reg mmxSize= srcWidth&~15;

        __asm__ volatile(

            "mov           %4, %%"REG_a"            \n\t"

            "movq        "MANGLE(mmx_ff)", %%mm0    \n\t"

            "movq         (%0, %%"REG_a"), %%mm4    \n\t"

            "movq                   %%mm4, %%mm2    \n\t"

            "psllq                     $8, %%mm4    \n\t"

            "pand                   %%mm0, %%mm2    \n\t"

            "por                    %%mm2, %%mm4    \n\t"

            "movq         (%1, %%"REG_a"), %%mm5    \n\t"

            "movq                   %%mm5, %%mm3    \n\t"

            "psllq                     $8, %%mm5    \n\t"

            "pand                   %%mm0, %%mm3    \n\t"

            "por                    %%mm3, %%mm5    \n\t"

            "1:                                     \n\t"

            "movq         (%0, %%"REG_a"), %%mm0    \n\t"

            "movq         (%1, %%"REG_a"), %%mm1    \n\t"

            "movq        1(%0, %%"REG_a"), %%mm2    \n\t"

            "movq        1(%1, %%"REG_a"), %%mm3    \n\t"

            PAVGB"                  %%mm0, %%mm5    \n\t"

            PAVGB"                  %%mm0, %%mm3    \n\t"

            PAVGB"                  %%mm0, %%mm5    \n\t"

            PAVGB"                  %%mm0, %%mm3    \n\t"

            PAVGB"                  %%mm1, %%mm4    \n\t"

            PAVGB"                  %%mm1, %%mm2    \n\t"

            PAVGB"                  %%mm1, %%mm4    \n\t"

            PAVGB"                  %%mm1, %%mm2    \n\t"

            "movq                   %%mm5, %%mm7    \n\t"

            "movq                   %%mm4, %%mm6    \n\t"

            "punpcklbw              %%mm3, %%mm5    \n\t"

            "punpckhbw              %%mm3, %%mm7    \n\t"

            "punpcklbw              %%mm2, %%mm4    \n\t"

            "punpckhbw              %%mm2, %%mm6    \n\t"

            MOVNTQ"                 %%mm5,  (%2, %%"REG_a", 2)  \n\t"

            MOVNTQ"                 %%mm7, 8(%2, %%"REG_a", 2)  \n\t"

            MOVNTQ"                 %%mm4,  (%3, %%"REG_a", 2)  \n\t"

            MOVNTQ"                 %%mm6, 8(%3, %%"REG_a", 2)  \n\t"

            "add                       $8, %%"REG_a"            \n\t"

            "movq       -1(%0, %%"REG_a"), %%mm4    \n\t"

            "movq       -1(%1, %%"REG_a"), %%mm5    \n\t"

            " js                       1b                       \n\t"

            :: "r" (src + mmxSize  ), "r" (src + srcStride + mmxSize  ),

               "r" (dst + mmxSize*2), "r" (dst + dstStride + mmxSize*2),

               "g" (-mmxSize)

               NAMED_CONSTRAINTS_ADD(mmx_ff)

            : "%"REG_a

        );



        for (x=mmxSize-1; x<srcWidth-1; x++) {

            dst[2*x          +1]= (3*src[x+0] +   src[x+srcStride+1])>>2;

            dst[2*x+dstStride+2]= (  src[x+0] + 3*src[x+srcStride+1])>>2;

            dst[2*x+dstStride+1]= (  src[x+1] + 3*src[x+srcStride  ])>>2;

            dst[2*x          +2]= (3*src[x+1] +   src[x+srcStride  ])>>2;

        }

        dst[srcWidth*2 -1            ]= (3*src[srcWidth-1] +   src[srcWidth-1 + srcStride])>>2;

        dst[srcWidth*2 -1 + dstStride]= (  src[srcWidth-1] + 3*src[srcWidth-1 + srcStride])>>2;



        dst+=dstStride*2;

        src+=srcStride;

    }



    // last line

    dst[0]= src[0];



    for (x=0; x<srcWidth-1; x++) {

        dst[2*x+1]= (3*src[x] +   src[x+1])>>2;

        dst[2*x+2]= (  src[x] + 3*src[x+1])>>2;

    }

    dst[2*srcWidth-1]= src[srcWidth-1];



    __asm__ volatile(EMMS"       \n\t"

                     SFENCE"     \n\t"

                     :::"memory");

}
