uint32_t helper_bcdsetsgn(ppc_avr_t *r, ppc_avr_t *b, uint32_t ps)

{

    int i;

    int invalid = 0;

    int sgnb = bcd_get_sgn(b);



    *r = *b;

    bcd_put_digit(r, bcd_preferred_sgn(sgnb, ps), 0);



    for (i = 1; i < 32; i++) {

        bcd_get_digit(b, i, &invalid);

        if (unlikely(invalid)) {

            return CRF_SO;

        }

    }



    return bcd_cmp_zero(r);

}
