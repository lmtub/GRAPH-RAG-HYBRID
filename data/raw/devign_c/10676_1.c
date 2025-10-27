static float get_band_cost_ZERO_mips(struct AACEncContext *s,

                                     PutBitContext *pb, const float *in,

                                     const float *scaled, int size, int scale_idx,

                                     int cb, const float lambda, const float uplim,

                                     int *bits)

{

    int i;

    float cost = 0;



    for (i = 0; i < size; i += 4) {

        cost += in[i  ] * in[i  ];

        cost += in[i+1] * in[i+1];

        cost += in[i+2] * in[i+2];

        cost += in[i+3] * in[i+3];

    }

    if (bits)

        *bits = 0;

    return cost * lambda;

}
