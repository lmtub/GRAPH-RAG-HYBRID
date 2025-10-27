static void encode_exponents_blk_ch(uint8_t *exp,

                                    int nb_exps, int exp_strategy,

                                    uint8_t *num_exp_groups)

{

    int group_size, nb_groups, i, j, k, exp_min;



    group_size = exp_strategy + (exp_strategy == EXP_D45);

    *num_exp_groups = (nb_exps + (group_size * 3) - 4) / (3 * group_size);

    nb_groups = *num_exp_groups * 3;



    /* for each group, compute the minimum exponent */

    if (exp_strategy > EXP_D15) {

    k = 1;

    for (i = 1; i <= nb_groups; i++) {

        exp_min = exp[k];

        assert(exp_min >= 0 && exp_min <= 24);

        for (j = 1; j < group_size; j++) {

            if (exp[k+j] < exp_min)

                exp_min = exp[k+j];

        }

        exp[i] = exp_min;

        k += group_size;

    }

    }



    /* constraint for DC exponent */

    if (exp[0] > 15)

        exp[0] = 15;



    /* decrease the delta between each groups to within 2 so that they can be

       differentially encoded */

    for (i = 1; i <= nb_groups; i++)

        exp[i] = FFMIN(exp[i], exp[i-1] + 2);

    for (i = nb_groups-1; i >= 0; i--)

        exp[i] = FFMIN(exp[i], exp[i+1] + 2);



    /* now we have the exponent values the decoder will see */

    if (exp_strategy > EXP_D15) {

    k = nb_groups * group_size;

    for (i = nb_groups; i > 0; i--) {

        for (j = 0; j < group_size; j++)

            exp[k-j] = exp[i];

        k -= group_size;

    }

    }

}
