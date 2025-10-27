static av_always_inline void decode_dc_coeffs(GetBitContext *gb, int16_t *out,

                                              int blocks_per_slice)

{

    int16_t prev_dc;

    int code, i, sign;



    OPEN_READER(re, gb);



    DECODE_CODEWORD(code, FIRST_DC_CB);

    prev_dc = TOSIGNED(code);

    out[0] = prev_dc;



    out += 64; // dc coeff for the next block



    code = 5;

    sign = 0;

    for (i = 1; i < blocks_per_slice; i++, out += 64) {

        DECODE_CODEWORD(code, dc_codebook[FFMIN(code, 6U)]);

        if(code) sign ^= -(code & 1);

        else     sign  = 0;

        prev_dc += (((code + 1) >> 1) ^ sign) - sign;

        out[0] = prev_dc;

    }

    CLOSE_READER(re, gb);

}
