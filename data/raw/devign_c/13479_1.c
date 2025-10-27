void float_to_int16_vfp(int16_t *dst, const float *src, int len)

{

    asm volatile(

        "fldmias    %[src]!, {s16-s23}\n\t"

        "ftosis     s0, s16\n\t"

        "ftosis     s1, s17\n\t"

        "ftosis     s2, s18\n\t"

        "ftosis     s3, s19\n\t"

        "ftosis     s4, s20\n\t"

        "ftosis     s5, s21\n\t"

        "ftosis     s6, s22\n\t"

        "ftosis     s7, s23\n\t"

    "1:\n\t"

        "subs       %[len], %[len], #8\n\t"

        "fmrrs      r3, r4, {s0, s1}\n\t"

        "fmrrs      r5, r6, {s2, s3}\n\t"

        "fmrrs      r7, r8, {s4, s5}\n\t"

        "fmrrs      ip, lr, {s6, s7}\n\t"

        "fldmiasgt  %[src]!, {s16-s23}\n\t"

        "ssat       r4, #16, r4\n\t"

        "ssat       r3, #16, r3\n\t"

        "ssat       r6, #16, r6\n\t"

        "ssat       r5, #16, r5\n\t"

        "pkhbt      r3, r3, r4, lsl #16\n\t"

        "pkhbt      r4, r5, r6, lsl #16\n\t"

        "ftosisgt   s0, s16\n\t"

        "ftosisgt   s1, s17\n\t"

        "ftosisgt   s2, s18\n\t"

        "ftosisgt   s3, s19\n\t"

        "ftosisgt   s4, s20\n\t"

        "ftosisgt   s5, s21\n\t"

        "ftosisgt   s6, s22\n\t"

        "ftosisgt   s7, s23\n\t"

        "ssat       r8, #16, r8\n\t"

        "ssat       r7, #16, r7\n\t"

        "ssat       lr, #16, lr\n\t"

        "ssat       ip, #16, ip\n\t"

        "pkhbt      r5, r7, r8, lsl #16\n\t"

        "pkhbt      r6, ip, lr, lsl #16\n\t"

        "stmia      %[dst]!, {r3-r6}\n\t"

        "bgt        1b\n\t"



        : [dst] "+&r" (dst), [src] "+&r" (src), [len] "+&r" (len)

        :

        : "s0",  "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",

          "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",

          "r3", "r4", "r5", "r6", "r7", "r8", "ip", "lr",

          "cc", "memory");

}
