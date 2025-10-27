static void DEF(put, pixels8_x2)(uint8_t *block, const uint8_t *pixels, ptrdiff_t line_size, int h)

{

    MOVQ_BFE(mm6);

    __asm__ volatile(

        "lea    (%3, %3), %%"REG_a"     \n\t"

        ".p2align 3                     \n\t"

        "1:                             \n\t"

        "movq   (%1), %%mm0             \n\t"

        "movq   1(%1), %%mm1            \n\t"

        "movq   (%1, %3), %%mm2         \n\t"

        "movq   1(%1, %3), %%mm3        \n\t"

        PAVGBP(%%mm0, %%mm1, %%mm4,   %%mm2, %%mm3, %%mm5)

        "movq   %%mm4, (%2)             \n\t"

        "movq   %%mm5, (%2, %3)         \n\t"

        "add    %%"REG_a", %1           \n\t"

        "add    %%"REG_a", %2           \n\t"

        "movq   (%1), %%mm0             \n\t"

        "movq   1(%1), %%mm1            \n\t"

        "movq   (%1, %3), %%mm2         \n\t"

        "movq   1(%1, %3), %%mm3        \n\t"

        PAVGBP(%%mm0, %%mm1, %%mm4,   %%mm2, %%mm3, %%mm5)

        "movq   %%mm4, (%2)             \n\t"

        "movq   %%mm5, (%2, %3)         \n\t"

        "add    %%"REG_a", %1           \n\t"

        "add    %%"REG_a", %2           \n\t"

        "subl   $4, %0                  \n\t"

        "jnz    1b                      \n\t"

        :"+g"(h), "+S"(pixels), "+D"(block)

        :"r"((x86_reg)line_size)

        :REG_a, "memory");

}
