static inline void RENAME(BEToUV)(uint8_t *dstU, uint8_t *dstV, const uint8_t *src1, const uint8_t *src2, long width, uint32_t *unused)

{

#if COMPILE_TEMPLATE_MMX

    __asm__ volatile(

        "movq "MANGLE(bm01010101)", %%mm4           \n\t"

        "mov                    %0, %%"REG_a"       \n\t"

        "1:                                         \n\t"

        "movq    (%1, %%"REG_a",2), %%mm0           \n\t"

        "movq   8(%1, %%"REG_a",2), %%mm1           \n\t"

        "movq    (%2, %%"REG_a",2), %%mm2           \n\t"

        "movq   8(%2, %%"REG_a",2), %%mm3           \n\t"

        "pand                %%mm4, %%mm0           \n\t"

        "pand                %%mm4, %%mm1           \n\t"

        "pand                %%mm4, %%mm2           \n\t"

        "pand                %%mm4, %%mm3           \n\t"

        "packuswb            %%mm1, %%mm0           \n\t"

        "packuswb            %%mm3, %%mm2           \n\t"

        "movq                %%mm0, (%3, %%"REG_a") \n\t"

        "movq                %%mm2, (%4, %%"REG_a") \n\t"

        "add                    $8, %%"REG_a"       \n\t"

        " js                    1b                  \n\t"

        : : "g" ((x86_reg)-width), "r" (src1+width*2), "r" (src2+width*2), "r" (dstU+width), "r" (dstV+width)

        : "%"REG_a

    );

#else

    int i;

    for (i=0; i<width; i++) {

        dstU[i]= src1[2*i];

        dstV[i]= src2[2*i];

    }

#endif

}
