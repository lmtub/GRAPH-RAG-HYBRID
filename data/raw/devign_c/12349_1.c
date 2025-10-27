static void vp5_parse_coeff(VP56Context *s)

{

    VP56RangeCoder *c = &s->c;

    VP56Model *model = s->modelp;

    uint8_t *permute = s->idct_scantable;

    uint8_t *model1, *model2;

    int coeff, sign, coeff_idx;

    int b, i, cg, idx, ctx, ctx_last;

    int pt = 0;    /* plane type (0 for Y, 1 for U or V) */



    for (b=0; b<6; b++) {

        int ct = 1;    /* code type */



        if (b > 3) pt = 1;



        ctx = 6*s->coeff_ctx[ff_vp56_b6to4[b]][0]

              + s->above_blocks[s->above_block_idx[b]].not_null_dc;

        model1 = model->coeff_dccv[pt];

        model2 = model->coeff_dcct[pt][ctx];



        coeff_idx = 0;

        for (;;) {

            if (vp56_rac_get_prob_branchy(c, model2[0])) {

                if (vp56_rac_get_prob_branchy(c, model2[2])) {

                    if (vp56_rac_get_prob_branchy(c, model2[3])) {

                        s->coeff_ctx[ff_vp56_b6to4[b]][coeff_idx] = 4;

                        idx = vp56_rac_get_tree(c, ff_vp56_pc_tree, model1);

                        sign = vp56_rac_get(c);

                        coeff = ff_vp56_coeff_bias[idx+5];

                        for (i=ff_vp56_coeff_bit_length[idx]; i>=0; i--)

                            coeff += vp56_rac_get_prob(c, ff_vp56_coeff_parse_table[idx][i]) << i;

                    } else {

                        if (vp56_rac_get_prob_branchy(c, model2[4])) {

                            coeff = 3 + vp56_rac_get_prob(c, model1[5]);

                            s->coeff_ctx[ff_vp56_b6to4[b]][coeff_idx] = 3;

                        } else {

                            coeff = 2;

                            s->coeff_ctx[ff_vp56_b6to4[b]][coeff_idx] = 2;

                        }

                        sign = vp56_rac_get(c);

                    }

                    ct = 2;

                } else {

                    ct = 1;

                    s->coeff_ctx[ff_vp56_b6to4[b]][coeff_idx] = 1;

                    sign = vp56_rac_get(c);

                    coeff = 1;

                }

                coeff = (coeff ^ -sign) + sign;

                if (coeff_idx)

                    coeff *= s->dequant_ac;

                s->block_coeff[b][permute[coeff_idx]] = coeff;

            } else {

                if (ct && !vp56_rac_get_prob_branchy(c, model2[1]))

                    break;

                ct = 0;

                s->coeff_ctx[ff_vp56_b6to4[b]][coeff_idx] = 0;

            }

            coeff_idx++;

            if (coeff_idx >= 64)

                break;



            cg = vp5_coeff_groups[coeff_idx];

            ctx = s->coeff_ctx[ff_vp56_b6to4[b]][coeff_idx];

            model1 = model->coeff_ract[pt][ct][cg];

            model2 = cg > 2 ? model1 : model->coeff_acct[pt][ct][cg][ctx];

        }



        ctx_last = FFMIN(s->coeff_ctx_last[ff_vp56_b6to4[b]], 24);

        s->coeff_ctx_last[ff_vp56_b6to4[b]] = coeff_idx;

        if (coeff_idx < ctx_last)

            for (i=coeff_idx; i<=ctx_last; i++)

                s->coeff_ctx[ff_vp56_b6to4[b]][i] = 5;

        s->above_blocks[s->above_block_idx[b]].not_null_dc = s->coeff_ctx[ff_vp56_b6to4[b]][0];

    }

}
