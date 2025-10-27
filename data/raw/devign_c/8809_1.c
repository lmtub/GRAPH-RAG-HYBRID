static void vp6_parse_coeff(VP56Context *s)

{

    VP56RangeCoder *c = s->ccp;

    VP56Model *model = s->modelp;

    uint8_t *permute = s->scantable.permutated;

    uint8_t *model1, *model2, *model3;

    int coeff, sign, coeff_idx;

    int b, i, cg, idx, ctx;

    int pt = 0;    /* plane type (0 for Y, 1 for U or V) */



    for (b=0; b<6; b++) {

        int ct = 1;    /* code type */

        int run = 1;



        if (b > 3) pt = 1;



        ctx = s->left_block[vp56_b6to4[b]].not_null_dc

              + s->above_blocks[s->above_block_idx[b]].not_null_dc;

        model1 = model->coeff_dccv[pt];

        model2 = model->coeff_dcct[pt][ctx];



        for (coeff_idx=0; coeff_idx<64; ) {

            if ((coeff_idx>1 && ct==0) || vp56_rac_get_prob(c, model2[0])) {

                /* parse a coeff */

                if (vp56_rac_get_prob(c, model2[2])) {

                    if (vp56_rac_get_prob(c, model2[3])) {

                        idx = vp56_rac_get_tree(c, vp56_pc_tree, model1);

                        coeff = vp56_coeff_bias[idx+5];

                        for (i=vp56_coeff_bit_length[idx]; i>=0; i--)

                            coeff += vp56_rac_get_prob(c, vp56_coeff_parse_table[idx][i]) << i;

                    } else {

                        if (vp56_rac_get_prob(c, model2[4]))

                            coeff = 3 + vp56_rac_get_prob(c, model1[5]);

                        else

                            coeff = 2;

                    }

                    ct = 2;

                } else {

                    ct = 1;

                    coeff = 1;

                }

                sign = vp56_rac_get(c);

                coeff = (coeff ^ -sign) + sign;

                if (coeff_idx)

                    coeff *= s->dequant_ac;

                idx = model->coeff_index_to_pos[coeff_idx];

                s->block_coeff[b][permute[idx]] = coeff;

                run = 1;

            } else {

                /* parse a run */

                ct = 0;

                if (coeff_idx > 0) {

                    if (!vp56_rac_get_prob(c, model2[1]))

                        break;



                    model3 = model->coeff_runv[coeff_idx >= 6];

                    run = vp56_rac_get_tree(c, vp6_pcr_tree, model3);

                    if (!run)

                        for (run=9, i=0; i<6; i++)

                            run += vp56_rac_get_prob(c, model3[i+8]) << i;

                }

            }



            cg = vp6_coeff_groups[coeff_idx+=run];

            model1 = model2 = model->coeff_ract[pt][ct][cg];

        }



        s->left_block[vp56_b6to4[b]].not_null_dc =

        s->above_blocks[s->above_block_idx[b]].not_null_dc = !!s->block_coeff[b][0];

    }

}
