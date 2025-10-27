static inline void encode_vlc_codeword(PutBitContext *pb, unsigned codebook, int val)

{

    unsigned int rice_order, exp_order, switch_bits, switch_val;

    int exponent;



    /* number of prefix bits to switch between Rice and expGolomb */

    switch_bits = (codebook & 3) + 1;

    rice_order  =  codebook >> 5;       /* rice code order */

    exp_order   = (codebook >> 2) & 7;  /* exp golomb code order */



    switch_val  = switch_bits << rice_order;



    if (val >= switch_val) {

        val -= switch_val - (1 << exp_order);

        exponent = av_log2(val);



        put_bits(pb, exponent - exp_order + switch_bits, 0);

        put_bits(pb, 1, 1);

        put_bits(pb, exponent, val);

    } else {

        exponent = val >> rice_order;



        if (exponent)

            put_bits(pb, exponent, 0);

        put_bits(pb, 1, 1);

        if (rice_order)

            put_sbits(pb, rice_order, val);

    }

}
