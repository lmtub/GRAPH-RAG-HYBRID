void ff_put_pixels_clamped_mmx(const DCTELEM *block, uint8_t *pixels,

                               int line_size)

{

    const DCTELEM *p;

    uint8_t *pix;



    /* read the pixels */

    p   = block;

    pix = pixels;

    /* unrolled loop */

    __asm__ volatile (

        "movq        %3, %%mm0          \n\t"

        "movq       8%3, %%mm1          \n\t"

        "movq      16%3, %%mm2          \n\t"

        "movq      24%3, %%mm3          \n\t"

        "movq      32%3, %%mm4          \n\t"

        "movq      40%3, %%mm5          \n\t"

        "movq      48%3, %%mm6          \n\t"

        "movq      56%3, %%mm7          \n\t"

        "packuswb %%mm1, %%mm0          \n\t"

        "packuswb %%mm3, %%mm2          \n\t"

        "packuswb %%mm5, %%mm4          \n\t"

        "packuswb %%mm7, %%mm6          \n\t"

        "movq     %%mm0, (%0)           \n\t"

        "movq     %%mm2, (%0, %1)       \n\t"

        "movq     %%mm4, (%0, %1, 2)    \n\t"

        "movq     %%mm6, (%0, %2)       \n\t"

        :: "r"(pix), "r"((x86_reg)line_size), "r"((x86_reg)line_size * 3),

           "m"(*p)

        : "memory");

    pix += line_size * 4;

    p   += 32;



    // if here would be an exact copy of the code above

    // compiler would generate some very strange code

    // thus using "r"

    __asm__ volatile (

        "movq       (%3), %%mm0         \n\t"

        "movq      8(%3), %%mm1         \n\t"

        "movq     16(%3), %%mm2         \n\t"

        "movq     24(%3), %%mm3         \n\t"

        "movq     32(%3), %%mm4         \n\t"

        "movq     40(%3), %%mm5         \n\t"

        "movq     48(%3), %%mm6         \n\t"

        "movq     56(%3), %%mm7         \n\t"

        "packuswb  %%mm1, %%mm0         \n\t"

        "packuswb  %%mm3, %%mm2         \n\t"

        "packuswb  %%mm5, %%mm4         \n\t"

        "packuswb  %%mm7, %%mm6         \n\t"

        "movq      %%mm0, (%0)          \n\t"

        "movq      %%mm2, (%0, %1)      \n\t"

        "movq      %%mm4, (%0, %1, 2)   \n\t"

        "movq      %%mm6, (%0, %2)      \n\t"

        :: "r"(pix), "r"((x86_reg)line_size), "r"((x86_reg)line_size * 3), "r"(p)

        : "memory");

}
