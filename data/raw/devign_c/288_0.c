static void group_exponents(AC3EncodeContext *s)

{

    int blk, ch, i;

    int group_size, nb_groups, bit_count;

    uint8_t *p;

    int delta0, delta1, delta2;

    int exp0, exp1;



    bit_count = 0;

    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {

        AC3Block *block = &s->blocks[blk];

        for (ch = 0; ch < s->channels; ch++) {

            if (s->exp_strategy[ch][blk] == EXP_REUSE)

                continue;

            group_size = s->exp_strategy[ch][blk] + (s->exp_strategy[ch][blk] == EXP_D45);

            nb_groups = exponent_group_tab[s->exp_strategy[ch][blk]-1][s->nb_coefs[ch]];

            bit_count += 4 + (nb_groups * 7);

            p = block->exp[ch];



            /* DC exponent */

            exp1 = *p++;

            block->grouped_exp[ch][0] = exp1;



            /* remaining exponents are delta encoded */

            for (i = 1; i <= nb_groups; i++) {

                /* merge three delta in one code */

                exp0   = exp1;

                exp1   = p[0];

                p     += group_size;

                delta0 = exp1 - exp0 + 2;



                exp0   = exp1;

                exp1   = p[0];

                p     += group_size;

                delta1 = exp1 - exp0 + 2;



                exp0   = exp1;

                exp1   = p[0];

                p     += group_size;

                delta2 = exp1 - exp0 + 2;



                block->grouped_exp[ch][i] = ((delta0 * 5 + delta1) * 5) + delta2;

            }

        }

    }



    s->exponent_bits = bit_count;

}
