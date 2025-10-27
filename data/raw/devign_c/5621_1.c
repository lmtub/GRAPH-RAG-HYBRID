static float get_band_cost_UPAIR7_mips(struct AACEncContext *s,

                                       PutBitContext *pb, const float *in,

                                       const float *scaled, int size, int scale_idx,

                                       int cb, const float lambda, const float uplim,

                                       int *bits)

{

    const float Q34 = ff_aac_pow34sf_tab[POW_SF2_ZERO - scale_idx + SCALE_ONE_POS - SCALE_DIV_512];

    const float IQ  = ff_aac_pow2sf_tab [POW_SF2_ZERO + scale_idx - SCALE_ONE_POS + SCALE_DIV_512];

    int i;

    float cost = 0;

    int qc1, qc2, qc3, qc4;

    int curbits = 0;



    uint8_t *p_bits  = (uint8_t *)ff_aac_spectral_bits[cb-1];

    float   *p_codes = (float   *)ff_aac_codebook_vectors[cb-1];



    for (i = 0; i < size; i += 4) {

        const float *vec, *vec2;

        int curidx, curidx2, sign1, count1, sign2, count2;

        int   *in_int = (int   *)&in[i];

        float *in_pos = (float *)&in[i];

        float di0, di1, di2, di3;

        int t0, t1, t2, t3, t4;



        qc1 = scaled[i  ] * Q34 + ROUND_STANDARD;

        qc2 = scaled[i+1] * Q34 + ROUND_STANDARD;

        qc3 = scaled[i+2] * Q34 + ROUND_STANDARD;

        qc4 = scaled[i+3] * Q34 + ROUND_STANDARD;



        __asm__ volatile (

            ".set push                                          \n\t"

            ".set noreorder                                     \n\t"



            "ori        %[t4],      $zero,      7               \n\t"

            "ori        %[sign1],   $zero,      0               \n\t"

            "ori        %[sign2],   $zero,      0               \n\t"

            "slt        %[t0],      %[t4],      %[qc1]          \n\t"

            "slt        %[t1],      %[t4],      %[qc2]          \n\t"

            "slt        %[t2],      %[t4],      %[qc3]          \n\t"

            "slt        %[t3],      %[t4],      %[qc4]          \n\t"

            "movn       %[qc1],     %[t4],      %[t0]           \n\t"

            "movn       %[qc2],     %[t4],      %[t1]           \n\t"

            "movn       %[qc3],     %[t4],      %[t2]           \n\t"

            "movn       %[qc4],     %[t4],      %[t3]           \n\t"

            "lw         %[t0],      0(%[in_int])                \n\t"

            "lw         %[t1],      4(%[in_int])                \n\t"

            "lw         %[t2],      8(%[in_int])                \n\t"

            "lw         %[t3],      12(%[in_int])               \n\t"

            "slt        %[t0],      %[t0],      $zero           \n\t"

            "movn       %[sign1],   %[t0],      %[qc1]          \n\t"

            "slt        %[t2],      %[t2],      $zero           \n\t"

            "movn       %[sign2],   %[t2],      %[qc3]          \n\t"

            "slt        %[t1],      %[t1],      $zero           \n\t"

            "sll        %[t0],      %[sign1],   1               \n\t"

            "or         %[t0],      %[t0],      %[t1]           \n\t"

            "movn       %[sign1],   %[t0],      %[qc2]          \n\t"

            "slt        %[t3],      %[t3],      $zero           \n\t"

            "sll        %[t0],      %[sign2],   1               \n\t"

            "or         %[t0],      %[t0],      %[t3]           \n\t"

            "movn       %[sign2],   %[t0],      %[qc4]          \n\t"

            "slt        %[count1],  $zero,      %[qc1]          \n\t"

            "slt        %[t1],      $zero,      %[qc2]          \n\t"

            "slt        %[count2],  $zero,      %[qc3]          \n\t"

            "slt        %[t2],      $zero,      %[qc4]          \n\t"

            "addu       %[count1],  %[count1],  %[t1]           \n\t"

            "addu       %[count2],  %[count2],  %[t2]           \n\t"



            ".set pop                                           \n\t"



            : [qc1]"+r"(qc1), [qc2]"+r"(qc2),

              [qc3]"+r"(qc3), [qc4]"+r"(qc4),

              [sign1]"=&r"(sign1), [count1]"=&r"(count1),

              [sign2]"=&r"(sign2), [count2]"=&r"(count2),

              [t0]"=&r"(t0), [t1]"=&r"(t1), [t2]"=&r"(t2), [t3]"=&r"(t3),

              [t4]"=&r"(t4)

            : [in_int]"r"(in_int)

            : "memory"

        );



        curidx = 8 * qc1;

        curidx += qc2;



        curidx2 = 8 * qc3;

        curidx2 += qc4;



        curbits += p_bits[curidx];

        curbits += upair7_sign_bits[curidx];

        vec     = &p_codes[curidx*2];



        curbits += p_bits[curidx2];

        curbits += upair7_sign_bits[curidx2];

        vec2    = &p_codes[curidx2*2];



        __asm__ volatile (

            ".set push                                          \n\t"

            ".set noreorder                                     \n\t"



            "lwc1       %[di0],     0(%[in_pos])                \n\t"

            "lwc1       %[di1],     4(%[in_pos])                \n\t"

            "lwc1       %[di2],     8(%[in_pos])                \n\t"

            "lwc1       %[di3],     12(%[in_pos])               \n\t"

            "abs.s      %[di0],     %[di0]                      \n\t"

            "abs.s      %[di1],     %[di1]                      \n\t"

            "abs.s      %[di2],     %[di2]                      \n\t"

            "abs.s      %[di3],     %[di3]                      \n\t"

            "lwc1       $f0,        0(%[vec])                   \n\t"

            "lwc1       $f1,        4(%[vec])                   \n\t"

            "lwc1       $f2,        0(%[vec2])                  \n\t"

            "lwc1       $f3,        4(%[vec2])                  \n\t"

            "nmsub.s    %[di0],     %[di0],     $f0,    %[IQ]   \n\t"

            "nmsub.s    %[di1],     %[di1],     $f1,    %[IQ]   \n\t"

            "nmsub.s    %[di2],     %[di2],     $f2,    %[IQ]   \n\t"

            "nmsub.s    %[di3],     %[di3],     $f3,    %[IQ]   \n\t"



            ".set pop                                           \n\t"



            : [di0]"=&f"(di0), [di1]"=&f"(di1),

              [di2]"=&f"(di2), [di3]"=&f"(di3)

            : [in_pos]"r"(in_pos), [vec]"r"(vec),

              [vec2]"r"(vec2), [IQ]"f"(IQ)

            : "$f0", "$f1", "$f2", "$f3",

              "memory"

        );



        cost += di0 * di0 + di1 * di1

                + di2 * di2 + di3 * di3;

    }



    if (bits)

        *bits = curbits;

    return cost * lambda + curbits;

}
