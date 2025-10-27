static void vector_fmul_reverse_vfp(float *dst, const float *src0, const float *src1, int len)

{

    src1 += len;

    asm volatile(

        "fldmdbs    %[src1]!, {s0-s3}\n\t"

        "fldmias    %[src0]!, {s8-s11}\n\t"

        "fldmdbs    %[src1]!, {s4-s7}\n\t"

        "fldmias    %[src0]!, {s12-s15}\n\t"

        "fmuls      s8, s3, s8\n\t"

        "fmuls      s9, s2, s9\n\t"

        "fmuls      s10, s1, s10\n\t"

        "fmuls      s11, s0, s11\n\t"

    "1:\n\t"

        "subs       %[len], %[len], #16\n\t"

        "fldmdbsge  %[src1]!, {s16-s19}\n\t"

        "fmuls      s12, s7, s12\n\t"

        "fldmiasge  %[src0]!, {s24-s27}\n\t"

        "fmuls      s13, s6, s13\n\t"

        "fldmdbsge  %[src1]!, {s20-s23}\n\t"

        "fmuls      s14, s5, s14\n\t"

        "fldmiasge  %[src0]!, {s28-s31}\n\t"

        "fmuls      s15, s4, s15\n\t"

        "fmulsge    s24, s19, s24\n\t"

        "fldmdbsgt  %[src1]!, {s0-s3}\n\t"

        "fmulsge    s25, s18, s25\n\t"

        "fstmias    %[dst]!, {s8-s13}\n\t"

        "fmulsge    s26, s17, s26\n\t"

        "fldmiasgt  %[src0]!, {s8-s11}\n\t"

        "fmulsge    s27, s16, s27\n\t"

        "fmulsge    s28, s23, s28\n\t"

        "fldmdbsgt  %[src1]!, {s4-s7}\n\t"

        "fmulsge    s29, s22, s29\n\t"

        "fstmias    %[dst]!, {s14-s15}\n\t"

        "fmulsge    s30, s21, s30\n\t"

        "fmulsge    s31, s20, s31\n\t"

        "fmulsge    s8, s3, s8\n\t"

        "fldmiasgt  %[src0]!, {s12-s15}\n\t"

        "fmulsge    s9, s2, s9\n\t"

        "fmulsge    s10, s1, s10\n\t"

        "fstmiasge  %[dst]!, {s24-s27}\n\t"

        "fmulsge    s11, s0, s11\n\t"

        "fstmiasge  %[dst]!, {s28-s31}\n\t"

        "bgt        1b\n\t"



        : [dst] "+&r" (dst), [src0] "+&r" (src0), [src1] "+&r" (src1), [len] "+&r" (len)

        :

        : "s0",  "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",

          "s8",  "s9",  "s10", "s11", "s12", "s13", "s14", "s15",

          "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",

          "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",

          "cc", "memory");

}
