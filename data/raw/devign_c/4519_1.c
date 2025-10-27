static inline void RENAME(rgb24tobgr15)(const uint8_t *src, uint8_t *dst, int src_size)

{

    const uint8_t *s = src;

    const uint8_t *end;

    const uint8_t *mm_end;

    uint16_t *d = (uint16_t *)dst;

    end = s + src_size;

    __asm__ volatile(PREFETCH"    %0"::"m"(*src):"memory");

    __asm__ volatile(

        "movq          %0, %%mm7    \n\t"

        "movq          %1, %%mm6    \n\t"

        ::"m"(red_15mask),"m"(green_15mask));

    mm_end = end - 11;

    while (s < mm_end) {

        __asm__ volatile(

            PREFETCH"    32%1           \n\t"

            "movd          %1, %%mm0    \n\t"

            "movd         3%1, %%mm3    \n\t"

            "punpckldq    6%1, %%mm0    \n\t"

            "punpckldq    9%1, %%mm3    \n\t"

            "movq       %%mm0, %%mm1    \n\t"

            "movq       %%mm0, %%mm2    \n\t"

            "movq       %%mm3, %%mm4    \n\t"

            "movq       %%mm3, %%mm5    \n\t"

            "psrlq         $3, %%mm0    \n\t"

            "psrlq         $3, %%mm3    \n\t"

            "pand          %2, %%mm0    \n\t"

            "pand          %2, %%mm3    \n\t"

            "psrlq         $6, %%mm1    \n\t"

            "psrlq         $6, %%mm4    \n\t"

            "pand       %%mm6, %%mm1    \n\t"

            "pand       %%mm6, %%mm4    \n\t"

            "psrlq         $9, %%mm2    \n\t"

            "psrlq         $9, %%mm5    \n\t"

            "pand       %%mm7, %%mm2    \n\t"

            "pand       %%mm7, %%mm5    \n\t"

            "por        %%mm1, %%mm0    \n\t"

            "por        %%mm4, %%mm3    \n\t"

            "por        %%mm2, %%mm0    \n\t"

            "por        %%mm5, %%mm3    \n\t"

            "psllq        $16, %%mm3    \n\t"

            "por        %%mm3, %%mm0    \n\t"

            MOVNTQ"     %%mm0, %0       \n\t"

            :"=m"(*d):"m"(*s),"m"(blue_15mask):"memory");

        d += 4;

        s += 12;

    }

    __asm__ volatile(SFENCE:::"memory");

    __asm__ volatile(EMMS:::"memory");

    while (s < end) {

        const int b = *s++;

        const int g = *s++;

        const int r = *s++;

        *d++ = (b>>3) | ((g&0xF8)<<2) | ((r&0xF8)<<7);

    }

}
