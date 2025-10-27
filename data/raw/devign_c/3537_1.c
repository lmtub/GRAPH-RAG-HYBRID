static void vector_fmul_window_mips(float *dst, const float *src0,

                                    const float *src1, const float *win, int len)

{

    float * dst_j, *win_j, *src0_i, *src1_j, *dst_i, *win_i;

    float temp, temp1, temp2, temp3;

    float s0, s01, s1, s11;

    float wi, wi1, wi2, wi3;

    float wj, wj1, wj2, wj3;

    const float * lp_end = win + len;



    win_i  = (float *)win;

    win_j  = (float *)(win + 2 * len -1);

    src1_j = (float *)(src1 + len - 1);

    src0_i = (float *)src0;

    dst_i  = (float *)dst;

    dst_j  = (float *)(dst + 2 * len -1);



    /* loop unrolled 4 times */

    __asm__ volatile (

        "1:"

        "lwc1    %[s1],     0(%[src1_j])                \n\t"

        "lwc1    %[wi],     0(%[win_i])                 \n\t"

        "lwc1    %[wj],     0(%[win_j])                 \n\t"

        "lwc1    %[s11],   -4(%[src1_j])                \n\t"

        "lwc1    %[wi1],    4(%[win_i])                 \n\t"

        "lwc1    %[wj1],   -4(%[win_j])                 \n\t"

        "lwc1    %[s0],     0(%[src0_i])                \n\t"

        "lwc1    %[s01],    4(%[src0_i])                \n\t"

        "mul.s   %[temp],   %[s1],   %[wi]              \n\t"

        "mul.s   %[temp1],  %[s1],   %[wj]              \n\t"

        "mul.s   %[temp2],  %[s11],  %[wi1]             \n\t"

        "mul.s   %[temp3],  %[s11],  %[wj1]             \n\t"

        "lwc1    %[s1],    -8(%[src1_j])                \n\t"

        "lwc1    %[wi2],    8(%[win_i])                 \n\t"

        "lwc1    %[wj2],   -8(%[win_j])                 \n\t"

        "lwc1    %[s11],   -12(%[src1_j])               \n\t"

        "msub.s  %[temp],   %[temp],  %[s0],  %[wj]     \n\t"

        "madd.s  %[temp1],  %[temp1], %[s0],  %[wi]     \n\t"

        "msub.s  %[temp2],  %[temp2], %[s01], %[wj1]    \n\t"

        "madd.s  %[temp3],  %[temp3], %[s01], %[wi1]    \n\t"

        "lwc1    %[wi3],    12(%[win_i])                \n\t"

        "lwc1    %[wj3],   -12(%[win_j])                \n\t"

        "lwc1    %[s0],     8(%[src0_i])                \n\t"

        "lwc1    %[s01],    12(%[src0_i])               \n\t"

        "addiu   %[src1_j],-16                          \n\t"

        "addiu   %[win_i],  16                          \n\t"

        "addiu   %[win_j], -16                          \n\t"

        "addiu   %[src0_i], 16                          \n\t"

        "swc1    %[temp],   0(%[dst_i])                 \n\t" /* dst[i] = s0*wj - s1*wi; */

        "swc1    %[temp1],  0(%[dst_j])                 \n\t" /* dst[j] = s0*wi + s1*wj; */

        "swc1    %[temp2],  4(%[dst_i])                 \n\t" /* dst[i+1] = s01*wj1 - s11*wi1; */

        "swc1    %[temp3], -4(%[dst_j])                 \n\t" /* dst[j-1] = s01*wi1 + s11*wj1; */

        "mul.s   %[temp],   %[s1],    %[wi2]            \n\t"

        "mul.s   %[temp1],  %[s1],    %[wj2]            \n\t"

        "mul.s   %[temp2],  %[s11],   %[wi3]            \n\t"

        "mul.s   %[temp3],  %[s11],   %[wj3]            \n\t"

        "msub.s  %[temp],   %[temp],  %[s0],  %[wj2]    \n\t"

        "madd.s  %[temp1],  %[temp1], %[s0],  %[wi2]    \n\t"

        "msub.s  %[temp2],  %[temp2], %[s01], %[wj3]    \n\t"

        "madd.s  %[temp3],  %[temp3], %[s01], %[wi3]    \n\t"

        "swc1    %[temp],   8(%[dst_i])                 \n\t" /* dst[i+2] = s0*wj2 - s1*wi2; */

        "swc1    %[temp1], -8(%[dst_j])                 \n\t" /* dst[j-2] = s0*wi2 + s1*wj2; */

        "swc1    %[temp2],  12(%[dst_i])                \n\t" /* dst[i+2] = s01*wj3 - s11*wi3; */

        "swc1    %[temp3], -12(%[dst_j])                \n\t" /* dst[j-3] = s01*wi3 + s11*wj3; */

        "addiu   %[dst_i],  16                          \n\t"

        "addiu   %[dst_j], -16                          \n\t"

        "bne     %[win_i], %[lp_end], 1b                \n\t"

        : [temp]"=&f"(temp), [temp1]"=&f"(temp1), [temp2]"=&f"(temp2),

          [temp3]"=&f"(temp3), [src0_i]"+r"(src0_i), [win_i]"+r"(win_i),

          [src1_j]"+r"(src1_j), [win_j]"+r"(win_j), [dst_i]"+r"(dst_i),

          [dst_j]"+r"(dst_j), [s0] "=&f"(s0), [s01]"=&f"(s01), [s1] "=&f"(s1),

          [s11]"=&f"(s11), [wi] "=&f"(wi), [wj] "=&f"(wj), [wi2]"=&f"(wi2),

          [wj2]"=&f"(wj2), [wi3]"=&f"(wi3), [wj3]"=&f"(wj3), [wi1]"=&f"(wi1),

          [wj1]"=&f"(wj1)

        : [lp_end]"r"(lp_end)

        : "memory"

    );

}
