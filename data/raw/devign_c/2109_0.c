int ff_dirac_golomb_read_32bit(DiracGolombLUT *lut_ctx, const uint8_t *buf,

                               int bytes, uint8_t *_dst, int coeffs)

{

    int i, b, c_idx = 0;

    int32_t *dst = (int32_t *)_dst;

    DiracGolombLUT *future[4], *l = &lut_ctx[2*LUT_SIZE + buf[0]];

    INIT_RESIDUE(res);



    for (b = 1; b <= bytes; b++) {

        future[0] = &lut_ctx[buf[b]];

        future[1] = future[0] + 1*LUT_SIZE;

        future[2] = future[0] + 2*LUT_SIZE;

        future[3] = future[0] + 3*LUT_SIZE;



        if ((c_idx + 1) > coeffs)

            return c_idx;



        /* res_bits is a hint for better branch prediction */

        if (res_bits && l->sign) {

            int32_t coeff = 1;

            APPEND_RESIDUE(res, l->preamble);

            for (i = 0; i < (res_bits >> 1) - 1; i++) {

                coeff <<= 1;

                coeff |= (res >> (RSIZE_BITS - 2*i - 2)) & 1;

            }

            dst[c_idx++] = l->sign * (coeff - 1);

            SET_RESIDUE(res, 0, 0);

        }



        memcpy(&dst[c_idx], l->ready, LUT_BITS*sizeof(int32_t));

        c_idx += l->ready_num;



        APPEND_RESIDUE(res, l->leftover);



        l = future[l->need_s ? 3 : !res_bits ? 2 : res_bits & 1];

    }



    return c_idx;

}
