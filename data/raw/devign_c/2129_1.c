static inline void RENAME(rgb24tobgr32)(const uint8_t *src, uint8_t *dst, int src_size)

{

    uint8_t *dest = dst;

    const uint8_t *s = src;

    const uint8_t *end;

    const uint8_t *mm_end;

    end = s + src_size;

    __asm__ volatile(PREFETCH"    %0"::"m"(*s):"memory");

    mm_end = end - 23;

    __asm__ volatile("movq        %0, %%mm7"::"m"(mask32a):"memory");

    while (s < mm_end) {

        __asm__ volatile(

            PREFETCH"    32%1           \n\t"

            "movd          %1, %%mm0    \n\t"

            "punpckldq    3%1, %%mm0    \n\t"

            "movd         6%1, %%mm1    \n\t"

            "punpckldq    9%1, %%mm1    \n\t"

            "movd        12%1, %%mm2    \n\t"

            "punpckldq   15%1, %%mm2    \n\t"

            "movd        18%1, %%mm3    \n\t"

            "punpckldq   21%1, %%mm3    \n\t"

            "por        %%mm7, %%mm0    \n\t"

            "por        %%mm7, %%mm1    \n\t"

            "por        %%mm7, %%mm2    \n\t"

            "por        %%mm7, %%mm3    \n\t"

            MOVNTQ"     %%mm0,   %0     \n\t"

            MOVNTQ"     %%mm1,  8%0     \n\t"

            MOVNTQ"     %%mm2, 16%0     \n\t"

            MOVNTQ"     %%mm3, 24%0"

            :"=m"(*dest)

            :"m"(*s)

            :"memory");

        dest += 32;

        s += 24;

    }

    __asm__ volatile(SFENCE:::"memory");

    __asm__ volatile(EMMS:::"memory");

    while (s < end) {

        *dest++ = *s++;

        *dest++ = *s++;

        *dest++ = *s++;

        *dest++ = 255;

    }

}
