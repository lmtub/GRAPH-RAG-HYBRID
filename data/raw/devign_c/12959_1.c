static inline void RENAME(nvXXtoUV)(uint8_t *dst1, uint8_t *dst2,

                                    const uint8_t *src, int width)

{

#if COMPILE_TEMPLATE_MMX

    __asm__ volatile(

        "movq "MANGLE(bm01010101)", %%mm4           \n\t"

        "mov                    %0, %%"REG_a"       \n\t"

        "1:                                         \n\t"

        "movq    (%1, %%"REG_a",2), %%mm0           \n\t"

        "movq   8(%1, %%"REG_a",2), %%mm1           \n\t"

        "movq                %%mm0, %%mm2           \n\t"

        "movq                %%mm1, %%mm3           \n\t"

        "pand                %%mm4, %%mm0           \n\t"

        "pand                %%mm4, %%mm1           \n\t"

        "psrlw                  $8, %%mm2           \n\t"

        "psrlw                  $8, %%mm3           \n\t"

        "packuswb            %%mm1, %%mm0           \n\t"

        "packuswb            %%mm3, %%mm2           \n\t"

        "movq                %%mm0, (%2, %%"REG_a") \n\t"

        "movq                %%mm2, (%3, %%"REG_a") \n\t"

        "add                    $8, %%"REG_a"       \n\t"

        " js                    1b                  \n\t"

        : : "g" ((x86_reg)-width), "r" (src+width*2), "r" (dst1+width), "r" (dst2+width)

        : "%"REG_a

    );

#else

    int i;

    for (i = 0; i < width; i++) {

        dst1[i] = src[2*i+0];

        dst2[i] = src[2*i+1];

    }

#endif

}
