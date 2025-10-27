static inline void RENAME(uyvyToY)(uint8_t *dst, const uint8_t *src, int width, uint32_t *unused)

{

#if COMPILE_TEMPLATE_MMX

    __asm__ volatile(

        "mov                  %0, %%"REG_a"         \n\t"

        "1:                                         \n\t"

        "movq  (%1, %%"REG_a",2), %%mm0             \n\t"

        "movq 8(%1, %%"REG_a",2), %%mm1             \n\t"

        "psrlw                $8, %%mm0             \n\t"

        "psrlw                $8, %%mm1             \n\t"

        "packuswb          %%mm1, %%mm0             \n\t"

        "movq              %%mm0, (%2, %%"REG_a")   \n\t"

        "add                  $8, %%"REG_a"         \n\t"

        " js                  1b                    \n\t"

        : : "g" ((x86_reg)-width), "r" (src+width*2), "r" (dst+width)

        : "%"REG_a

    );

#else

    int i;

    for (i=0; i<width; i++)

        dst[i]= src[2*i+1];

#endif

}
