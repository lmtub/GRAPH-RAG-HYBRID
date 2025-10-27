static void RENAME(extract_odd2)(const uint8_t *src, uint8_t *dst0, uint8_t *dst1, x86_reg count)

{

    dst0+=   count;

    dst1+=   count;

    src += 4*count;

    count= - count;

#if COMPILE_TEMPLATE_MMX

    if(count <= -8) {

        count += 7;

        __asm__ volatile(

            "pcmpeqw       %%mm7, %%mm7        \n\t"

            "psrlw            $8, %%mm7        \n\t"

            "1:                                \n\t"

            "movq -28(%1, %0, 4), %%mm0        \n\t"

            "movq -20(%1, %0, 4), %%mm1        \n\t"

            "movq -12(%1, %0, 4), %%mm2        \n\t"

            "movq  -4(%1, %0, 4), %%mm3        \n\t"

            "psrlw            $8, %%mm0        \n\t"

            "psrlw            $8, %%mm1        \n\t"

            "psrlw            $8, %%mm2        \n\t"

            "psrlw            $8, %%mm3        \n\t"

            "packuswb      %%mm1, %%mm0        \n\t"

            "packuswb      %%mm3, %%mm2        \n\t"

            "movq          %%mm0, %%mm1        \n\t"

            "movq          %%mm2, %%mm3        \n\t"

            "psrlw            $8, %%mm0        \n\t"

            "psrlw            $8, %%mm2        \n\t"

            "pand          %%mm7, %%mm1        \n\t"

            "pand          %%mm7, %%mm3        \n\t"

            "packuswb      %%mm2, %%mm0        \n\t"

            "packuswb      %%mm3, %%mm1        \n\t"

            MOVNTQ"        %%mm0,- 7(%3, %0)   \n\t"

            MOVNTQ"        %%mm1,- 7(%2, %0)   \n\t"

            "add              $8, %0           \n\t"

            " js 1b                            \n\t"

            : "+r"(count)

            : "r"(src), "r"(dst0), "r"(dst1)

        );

        count -= 7;

    }

#endif

    src++;

    while(count<0) {

        dst0[count]= src[4*count+0];

        dst1[count]= src[4*count+2];

        count++;

    }

}
