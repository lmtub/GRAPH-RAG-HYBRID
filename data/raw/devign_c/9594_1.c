static void vector_fmul_vfp(float *dst, const float *src, int len)

{

    int tmp;

    asm volatile(

        "fmrx       %[tmp], fpscr\n\t"

        "orr        %[tmp], %[tmp], #(3 << 16)\n\t" /* set vector size to 4 */

        "fmxr       fpscr, %[tmp]\n\t"



        "fldmias    %[dst_r]!, {s0-s3}\n\t"

        "fldmias    %[src]!, {s8-s11}\n\t"

        "fldmias    %[dst_r]!, {s4-s7}\n\t"

        "fldmias    %[src]!, {s12-s15}\n\t"

        "fmuls      s8, s0, s8\n\t"

    "1:\n\t"

        "subs       %[len], %[len], #16\n\t"

        "fmuls      s12, s4, s12\n\t"

        "fldmiasge  %[dst_r]!, {s16-s19}\n\t"

        "fldmiasge  %[src]!, {s24-s27}\n\t"

        "fldmiasge  %[dst_r]!, {s20-s23}\n\t"

        "fldmiasge  %[src]!, {s28-s31}\n\t"

        "fmulsge    s24, s16, s24\n\t"

        "fstmias    %[dst_w]!, {s8-s11}\n\t"

        "fstmias    %[dst_w]!, {s12-s15}\n\t"

        "fmulsge    s28, s20, s28\n\t"

        "fldmiasgt  %[dst_r]!, {s0-s3}\n\t"

        "fldmiasgt  %[src]!, {s8-s11}\n\t"

        "fldmiasgt  %[dst_r]!, {s4-s7}\n\t"

        "fldmiasgt  %[src]!, {s12-s15}\n\t"

        "fmulsge    s8, s0, s8\n\t"

        "fstmiasge  %[dst_w]!, {s24-s27}\n\t"

        "fstmiasge  %[dst_w]!, {s28-s31}\n\t"

        "bgt        1b\n\t"



        "bic        %[tmp], %[tmp], #(7 << 16)\n\t" /* set vector size back to 1 */

        "fmxr       fpscr, %[tmp]\n\t"

        : [dst_w] "+&r" (dst), [dst_r] "+&r" (dst), [src] "+&r" (src), [len] "+&r" (len), [tmp] "=&r" (tmp)

        :

        : "s0",  "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",

          "s8",  "s9",  "s10", "s11", "s12", "s13", "s14", "s15",

          "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",

          "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",

          "cc", "memory");

}
