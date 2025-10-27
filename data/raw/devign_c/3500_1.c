static float get_band_cost_SQUAD_mips(struct AACEncContext *s,

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

        const float *vec;

        int curidx;

        int   *in_int = (int   *)&in[i];

        float *in_pos = (float *)&in[i];

        float di0, di1, di2, di3;

        int t0, t1, t2, t3, t4, t5, t6, t7;



        qc1 = scaled[i  ] * Q34 + ROUND_STANDARD;

        qc2 = scaled[i+1] * Q34 + ROUND_STANDARD;

        qc3 = scaled[i+2] * Q34 + ROUND_STANDARD;

        qc4 = scaled[i+3] * Q34 + ROUND_STANDARD;



        __asm__ volatile (

            ".set push                                  \n\t"

            ".set noreorder                             \n\t"



            "slt        %[qc1], $zero,  %[qc1]          \n\t"

            "slt        %[qc2], $zero,  %[qc2]          \n\t"

            "slt        %[qc3], $zero,  %[qc3]          \n\t"

            "slt        %[qc4], $zero,  %[qc4]          \n\t"

            "lw         %[t0],  0(%[in_int])            \n\t"

            "lw         %[t1],  4(%[in_int])            \n\t"

            "lw         %[t2],  8(%[in_int])            \n\t"

            "lw         %[t3],  12(%[in_int])           \n\t"

            "srl        %[t0],  %[t0],  31              \n\t"

            "srl        %[t1],  %[t1],  31              \n\t"

            "srl        %[t2],  %[t2],  31              \n\t"

            "srl        %[t3],  %[t3],  31              \n\t"

            "subu       %[t4],  $zero,  %[qc1]          \n\t"

            "subu       %[t5],  $zero,  %[qc2]          \n\t"

            "subu       %[t6],  $zero,  %[qc3]          \n\t"

            "subu       %[t7],  $zero,  %[qc4]          \n\t"

            "movn       %[qc1], %[t4],  %[t0]           \n\t"

            "movn       %[qc2], %[t5],  %[t1]           \n\t"

            "movn       %[qc3], %[t6],  %[t2]           \n\t"

            "movn       %[qc4], %[t7],  %[t3]           \n\t"



            ".set pop                                   \n\t"



            : [qc1]"+r"(qc1), [qc2]"+r"(qc2),

              [qc3]"+r"(qc3), [qc4]"+r"(qc4),

              [t0]"=&r"(t0), [t1]"=&r"(t1), [t2]"=&r"(t2), [t3]"=&r"(t3),

              [t4]"=&r"(t4), [t5]"=&r"(t5), [t6]"=&r"(t6), [t7]"=&r"(t7)

            : [in_int]"r"(in_int)

            : "memory"

        );



        curidx = qc1;

        curidx *= 3;

        curidx += qc2;

        curidx *= 3;

        curidx += qc3;

        curidx *= 3;

        curidx += qc4;

        curidx += 40;



        curbits += p_bits[curidx];

        vec     = &p_codes[curidx*4];



        __asm__ volatile (

            ".set push                                  \n\t"

            ".set noreorder                             \n\t"



            "lwc1       $f0,    0(%[in_pos])            \n\t"

            "lwc1       $f1,    0(%[vec])               \n\t"

            "lwc1       $f2,    4(%[in_pos])            \n\t"

            "lwc1       $f3,    4(%[vec])               \n\t"

            "lwc1       $f4,    8(%[in_pos])            \n\t"

            "lwc1       $f5,    8(%[vec])               \n\t"

            "lwc1       $f6,    12(%[in_pos])           \n\t"

            "lwc1       $f7,    12(%[vec])              \n\t"

            "nmsub.s    %[di0], $f0,    $f1,    %[IQ]   \n\t"

            "nmsub.s    %[di1], $f2,    $f3,    %[IQ]   \n\t"

            "nmsub.s    %[di2], $f4,    $f5,    %[IQ]   \n\t"

            "nmsub.s    %[di3], $f6,    $f7,    %[IQ]   \n\t"



            ".set pop                                   \n\t"



            : [di0]"=&f"(di0), [di1]"=&f"(di1),

              [di2]"=&f"(di2), [di3]"=&f"(di3)

            : [in_pos]"r"(in_pos), [vec]"r"(vec),

              [IQ]"f"(IQ)

            : "$f0", "$f1", "$f2", "$f3",

              "$f4", "$f5", "$f6", "$f7",

              "memory"

        );



        cost += di0 * di0 + di1 * di1

                + di2 * di2 + di3 * di3;

    }



    if (bits)

        *bits = curbits;

    return cost * lambda + curbits;

}
