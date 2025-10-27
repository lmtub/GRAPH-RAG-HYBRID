static inline void RENAME(rgb16to32)(const uint8_t *src, uint8_t *dst, int src_size)

{

    const uint16_t *end;

    const uint16_t *mm_end;

    uint8_t *d = dst;

    const uint16_t *s = (const uint16_t*)src;

    end = s + src_size/2;

    __asm__ volatile(PREFETCH"    %0"::"m"(*s):"memory");

    __asm__ volatile("pxor    %%mm7,%%mm7    \n\t":::"memory");

    __asm__ volatile("pcmpeqd %%mm6,%%mm6    \n\t":::"memory");

    mm_end = end - 3;

    while (s < mm_end) {

        __asm__ volatile(

            PREFETCH"    32%1           \n\t"

            "movq          %1, %%mm0    \n\t"

            "movq          %1, %%mm1    \n\t"

            "movq          %1, %%mm2    \n\t"

            "pand          %2, %%mm0    \n\t"

            "pand          %3, %%mm1    \n\t"

            "pand          %4, %%mm2    \n\t"

            "psllq         $3, %%mm0    \n\t"

            "psrlq         $3, %%mm1    \n\t"

            "psrlq         $8, %%mm2    \n\t"

            PACK_RGB32

            :"=m"(*d)

            :"m"(*s),"m"(mask16b),"m"(mask16g),"m"(mask16r)

            :"memory");

        d += 16;

        s += 4;

    }

    __asm__ volatile(SFENCE:::"memory");

    __asm__ volatile(EMMS:::"memory");

    while (s < end) {

        register uint16_t bgr;

        bgr = *s++;

        *d++ = (bgr&0x1F)<<3;

        *d++ = (bgr&0x7E0)>>3;

        *d++ = (bgr&0xF800)>>8;

        *d++ = 255;

    }

}
