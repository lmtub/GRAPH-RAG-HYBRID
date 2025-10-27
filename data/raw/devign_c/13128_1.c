static inline void RENAME(rgb32tobgr15)(const uint8_t *src, uint8_t *dst, int src_size)

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

    mm_end = end - 15;

    while (s < mm_end) {

        __asm__ volatile(

            PREFETCH"    32%1           \n\t"

            "movd          %1, %%mm0    \n\t"

            "movd         4%1, %%mm3    \n\t"

            "punpckldq    8%1, %%mm0    \n\t"

            "punpckldq   12%1, %%mm3    \n\t"

            "movq       %%mm0, %%mm1    \n\t"

            "movq       %%mm0, %%mm2    \n\t"

            "movq       %%mm3, %%mm4    \n\t"

            "movq       %%mm3, %%mm5    \n\t"

            "psllq         $7, %%mm0    \n\t"

            "psllq         $7, %%mm3    \n\t"

            "pand       %%mm7, %%mm0    \n\t"

            "pand       %%mm7, %%mm3    \n\t"

            "psrlq         $6, %%mm1    \n\t"

            "psrlq         $6, %%mm4    \n\t"

            "pand       %%mm6, %%mm1    \n\t"

            "pand       %%mm6, %%mm4    \n\t"

            "psrlq        $19, %%mm2    \n\t"

            "psrlq        $19, %%mm5    \n\t"

            "pand          %2, %%mm2    \n\t"

            "pand          %2, %%mm5    \n\t"

            "por        %%mm1, %%mm0    \n\t"

            "por        %%mm4, %%mm3    \n\t"

            "por        %%mm2, %%mm0    \n\t"

            "por        %%mm5, %%mm3    \n\t"

            "psllq        $16, %%mm3    \n\t"

            "por        %%mm3, %%mm0    \n\t"

            MOVNTQ"     %%mm0, %0       \n\t"

            :"=m"(*d):"m"(*s),"m"(blue_15mask):"memory");

        d += 4;

        s += 16;

    }

    __asm__ volatile(SFENCE:::"memory");

    __asm__ volatile(EMMS:::"memory");

    while (s < end) {

        register int rgb = *(const uint32_t*)s; s += 4;

        *d++ = ((rgb&0xF8)<<7) + ((rgb&0xF800)>>6) + ((rgb&0xF80000)>>19);

    }

}
