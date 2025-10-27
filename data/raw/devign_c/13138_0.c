static void vc1_put_ver_16b_shift2_mmx(int16_t *dst,

                                       const uint8_t *src, x86_reg stride,

                                       int rnd, int64_t shift)

{

    __asm__ volatile(

        "mov       $3, %%"REG_c"           \n\t"

        LOAD_ROUNDER_MMX("%5")

        "movq      "MANGLE(ff_pw_9)", %%mm6 \n\t"

        "1:                                \n\t"

        "movd      (%0), %%mm2             \n\t"

        "add       %2, %0                  \n\t"

        "movd      (%0), %%mm3             \n\t"

        "punpcklbw %%mm0, %%mm2            \n\t"

        "punpcklbw %%mm0, %%mm3            \n\t"

        SHIFT2_LINE(  0, 1, 2, 3, 4)

        SHIFT2_LINE( 24, 2, 3, 4, 1)

        SHIFT2_LINE( 48, 3, 4, 1, 2)

        SHIFT2_LINE( 72, 4, 1, 2, 3)

        SHIFT2_LINE( 96, 1, 2, 3, 4)

        SHIFT2_LINE(120, 2, 3, 4, 1)

        SHIFT2_LINE(144, 3, 4, 1, 2)

        SHIFT2_LINE(168, 4, 1, 2, 3)

        "sub       %6, %0                  \n\t"

        "add       $8, %1                  \n\t"

        "dec       %%"REG_c"               \n\t"

        "jnz 1b                            \n\t"

        : "+r"(src), "+r"(dst)

        : "r"(stride), "r"(-2*stride),

          "m"(shift), "m"(rnd), "r"(9*stride-4)

          NAMED_CONSTRAINTS_ADD(ff_pw_9)

        : "%"REG_c, "memory"

    );

}
