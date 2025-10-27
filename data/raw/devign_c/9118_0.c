static inline void RENAME(hScale)(int16_t *dst, int dstW, const uint8_t *src, int srcW, int xInc,

                                  const int16_t *filter, const int16_t *filterPos, long filterSize)

{

#if COMPILE_TEMPLATE_MMX

    assert(filterSize % 4 == 0 && filterSize>0);

    if (filterSize==4) { // Always true for upscaling, sometimes for down, too.

        x86_reg counter= -2*dstW;

        filter-= counter*2;

        filterPos-= counter/2;

        dst-= counter/2;

        __asm__ volatile(

#if defined(PIC)

            "push            %%"REG_b"              \n\t"

#endif

            "pxor                %%mm7, %%mm7       \n\t"

            "push           %%"REG_BP"              \n\t" // we use 7 regs here ...

            "mov             %%"REG_a", %%"REG_BP"  \n\t"

            ".p2align                4              \n\t"

            "1:                                     \n\t"

            "movzwl   (%2, %%"REG_BP"), %%eax       \n\t"

            "movzwl  2(%2, %%"REG_BP"), %%ebx       \n\t"

            "movq  (%1, %%"REG_BP", 4), %%mm1       \n\t"

            "movq 8(%1, %%"REG_BP", 4), %%mm3       \n\t"

            "movd      (%3, %%"REG_a"), %%mm0       \n\t"

            "movd      (%3, %%"REG_b"), %%mm2       \n\t"

            "punpcklbw           %%mm7, %%mm0       \n\t"

            "punpcklbw           %%mm7, %%mm2       \n\t"

            "pmaddwd             %%mm1, %%mm0       \n\t"

            "pmaddwd             %%mm2, %%mm3       \n\t"

            "movq                %%mm0, %%mm4       \n\t"

            "punpckldq           %%mm3, %%mm0       \n\t"

            "punpckhdq           %%mm3, %%mm4       \n\t"

            "paddd               %%mm4, %%mm0       \n\t"

            "psrad                  $7, %%mm0       \n\t"

            "packssdw            %%mm0, %%mm0       \n\t"

            "movd                %%mm0, (%4, %%"REG_BP")    \n\t"

            "add                    $4, %%"REG_BP"  \n\t"

            " jnc                   1b              \n\t"



            "pop            %%"REG_BP"              \n\t"

#if defined(PIC)

            "pop             %%"REG_b"              \n\t"

#endif

            : "+a" (counter)

            : "c" (filter), "d" (filterPos), "S" (src), "D" (dst)

#if !defined(PIC)

            : "%"REG_b

#endif

        );

    } else if (filterSize==8) {

        x86_reg counter= -2*dstW;

        filter-= counter*4;

        filterPos-= counter/2;

        dst-= counter/2;

        __asm__ volatile(

#if defined(PIC)

            "push             %%"REG_b"             \n\t"

#endif

            "pxor                 %%mm7, %%mm7      \n\t"

            "push            %%"REG_BP"             \n\t" // we use 7 regs here ...

            "mov              %%"REG_a", %%"REG_BP" \n\t"

            ".p2align                 4             \n\t"

            "1:                                     \n\t"

            "movzwl    (%2, %%"REG_BP"), %%eax      \n\t"

            "movzwl   2(%2, %%"REG_BP"), %%ebx      \n\t"

            "movq   (%1, %%"REG_BP", 8), %%mm1      \n\t"

            "movq 16(%1, %%"REG_BP", 8), %%mm3      \n\t"

            "movd       (%3, %%"REG_a"), %%mm0      \n\t"

            "movd       (%3, %%"REG_b"), %%mm2      \n\t"

            "punpcklbw            %%mm7, %%mm0      \n\t"

            "punpcklbw            %%mm7, %%mm2      \n\t"

            "pmaddwd              %%mm1, %%mm0      \n\t"

            "pmaddwd              %%mm2, %%mm3      \n\t"



            "movq  8(%1, %%"REG_BP", 8), %%mm1      \n\t"

            "movq 24(%1, %%"REG_BP", 8), %%mm5      \n\t"

            "movd      4(%3, %%"REG_a"), %%mm4      \n\t"

            "movd      4(%3, %%"REG_b"), %%mm2      \n\t"

            "punpcklbw            %%mm7, %%mm4      \n\t"

            "punpcklbw            %%mm7, %%mm2      \n\t"

            "pmaddwd              %%mm1, %%mm4      \n\t"

            "pmaddwd              %%mm2, %%mm5      \n\t"

            "paddd                %%mm4, %%mm0      \n\t"

            "paddd                %%mm5, %%mm3      \n\t"

            "movq                 %%mm0, %%mm4      \n\t"

            "punpckldq            %%mm3, %%mm0      \n\t"

            "punpckhdq            %%mm3, %%mm4      \n\t"

            "paddd                %%mm4, %%mm0      \n\t"

            "psrad                   $7, %%mm0      \n\t"

            "packssdw             %%mm0, %%mm0      \n\t"

            "movd                 %%mm0, (%4, %%"REG_BP")   \n\t"

            "add                     $4, %%"REG_BP" \n\t"

            " jnc                    1b             \n\t"



            "pop             %%"REG_BP"             \n\t"

#if defined(PIC)

            "pop              %%"REG_b"             \n\t"

#endif

            : "+a" (counter)

            : "c" (filter), "d" (filterPos), "S" (src), "D" (dst)

#if !defined(PIC)

            : "%"REG_b

#endif

        );

    } else {

        const uint8_t *offset = src+filterSize;

        x86_reg counter= -2*dstW;

        //filter-= counter*filterSize/2;

        filterPos-= counter/2;

        dst-= counter/2;

        __asm__ volatile(

            "pxor                  %%mm7, %%mm7     \n\t"

            ".p2align                  4            \n\t"

            "1:                                     \n\t"

            "mov                      %2, %%"REG_c" \n\t"

            "movzwl      (%%"REG_c", %0), %%eax     \n\t"

            "movzwl     2(%%"REG_c", %0), %%edx     \n\t"

            "mov                      %5, %%"REG_c" \n\t"

            "pxor                  %%mm4, %%mm4     \n\t"

            "pxor                  %%mm5, %%mm5     \n\t"

            "2:                                     \n\t"

            "movq                   (%1), %%mm1     \n\t"

            "movq               (%1, %6), %%mm3     \n\t"

            "movd (%%"REG_c", %%"REG_a"), %%mm0     \n\t"

            "movd (%%"REG_c", %%"REG_d"), %%mm2     \n\t"

            "punpcklbw             %%mm7, %%mm0     \n\t"

            "punpcklbw             %%mm7, %%mm2     \n\t"

            "pmaddwd               %%mm1, %%mm0     \n\t"

            "pmaddwd               %%mm2, %%mm3     \n\t"

            "paddd                 %%mm3, %%mm5     \n\t"

            "paddd                 %%mm0, %%mm4     \n\t"

            "add                      $8, %1        \n\t"

            "add                      $4, %%"REG_c" \n\t"

            "cmp                      %4, %%"REG_c" \n\t"

            " jb                      2b            \n\t"

            "add                      %6, %1        \n\t"

            "movq                  %%mm4, %%mm0     \n\t"

            "punpckldq             %%mm5, %%mm4     \n\t"

            "punpckhdq             %%mm5, %%mm0     \n\t"

            "paddd                 %%mm0, %%mm4     \n\t"

            "psrad                    $7, %%mm4     \n\t"

            "packssdw              %%mm4, %%mm4     \n\t"

            "mov                      %3, %%"REG_a" \n\t"

            "movd                  %%mm4, (%%"REG_a", %0)   \n\t"

            "add                      $4, %0        \n\t"

            " jnc                     1b            \n\t"



            : "+r" (counter), "+r" (filter)

            : "m" (filterPos), "m" (dst), "m"(offset),

            "m" (src), "r" ((x86_reg)filterSize*2)

            : "%"REG_a, "%"REG_c, "%"REG_d

        );

    }

#else

#if COMPILE_TEMPLATE_ALTIVEC

    hScale_altivec_real(dst, dstW, src, srcW, xInc, filter, filterPos, filterSize);

#else

    int i;

    for (i=0; i<dstW; i++) {

        int j;

        int srcPos= filterPos[i];

        int val=0;

        //printf("filterPos: %d\n", filterPos[i]);

        for (j=0; j<filterSize; j++) {

            //printf("filter: %d, src: %d\n", filter[i], src[srcPos + j]);

            val += ((int)src[srcPos + j])*filter[filterSize*i + j];

        }

        //filter += hFilterSize;

        dst[i] = FFMIN(val>>7, (1<<15)-1); // the cubic equation does overflow ...

        //dst[i] = val>>7;

    }

#endif /* COMPILE_TEMPLATE_ALTIVEC */

#endif /* COMPILE_MMX */

}
