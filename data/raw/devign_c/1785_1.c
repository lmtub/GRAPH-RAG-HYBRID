static inline void RENAME(uyvyToUV)(uint8_t *dstU, uint8_t *dstV, const uint8_t *src1, const uint8_t *src2, int width, uint32_t *unused)

{

#if COMPILE_TEMPLATE_MMX

    __asm__ volatile(

        "movq "MANGLE(bm01010101)", %%mm4           \n\t"

        "mov                    %0, %%"REG_a"       \n\t"

        "1:                                         \n\t"

        "movq    (%1, %%"REG_a",4), %%mm0           \n\t"

        "movq   8(%1, %%"REG_a",4), %%mm1           \n\t"

        "pand                %%mm4, %%mm0           \n\t"

        "pand                %%mm4, %%mm1           \n\t"

        "packuswb            %%mm1, %%mm0           \n\t"

        "movq                %%mm0, %%mm1           \n\t"

        "psrlw                  $8, %%mm0           \n\t"

        "pand                %%mm4, %%mm1           \n\t"

        "packuswb            %%mm0, %%mm0           \n\t"

        "packuswb            %%mm1, %%mm1           \n\t"

        "movd                %%mm0, (%3, %%"REG_a") \n\t"

        "movd                %%mm1, (%2, %%"REG_a") \n\t"

        "add                    $4, %%"REG_a"       \n\t"

        " js                    1b                  \n\t"

        : : "g" ((x86_reg)-width), "r" (src1+width*4), "r" (dstU+width), "r" (dstV+width)

        : "%"REG_a

    );

#else

    int i;

    for (i=0; i<width; i++) {

        dstU[i]= src1[4*i + 0];

        dstV[i]= src1[4*i + 2];

    }

#endif

    assert(src1 == src2);

}
