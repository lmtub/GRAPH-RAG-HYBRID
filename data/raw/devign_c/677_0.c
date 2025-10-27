static inline void codeblock(DiracContext *s, SubBand *b,

                             GetBitContext *gb, DiracArith *c,

                             int left, int right, int top, int bottom,

                             int blockcnt_one, int is_arith)

{

    int x, y, zero_block;

    int qoffset, qfactor;

    IDWTELEM *buf;



    /* check for any coded coefficients in this codeblock */

    if (!blockcnt_one) {

        if (is_arith)

            zero_block = dirac_get_arith_bit(c, CTX_ZERO_BLOCK);

        else

            zero_block = get_bits1(gb);



        if (zero_block)

            return;

    }



    if (s->codeblock_mode && !(s->old_delta_quant && blockcnt_one)) {

        if (is_arith)

            b->quant += dirac_get_arith_int(c, CTX_DELTA_Q_F, CTX_DELTA_Q_DATA);

        else

            b->quant += dirac_get_se_golomb(gb);

    }



    b->quant = FFMIN(b->quant, MAX_QUANT);



    qfactor = qscale_tab[b->quant];

    /* TODO: context pointer? */

    if (!s->num_refs)

        qoffset = qoffset_intra_tab[b->quant];

    else

        qoffset = qoffset_inter_tab[b->quant];



    buf = b->ibuf + top * b->stride;

    for (y = top; y < bottom; y++) {

        for (x = left; x < right; x++) {

            /* [DIRAC_STD] 13.4.4 Subband coefficients. coeff_unpack() */

            if (is_arith)

                coeff_unpack_arith(c, qfactor, qoffset, b, buf+x, x, y);

            else

                buf[x] = coeff_unpack_golomb(gb, qfactor, qoffset);

        }

        buf += b->stride;

    }

}
