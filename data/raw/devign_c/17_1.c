static void vp6_parse_coeff_huffman(VP56Context *s)

{

    VP56Model *model = s->modelp;

    uint8_t *permute = s->scantable.permutated;

    VLC *vlc_coeff;

    int coeff, sign, coeff_idx;

    int b, cg, idx;

    int pt = 0;    /* plane type (0 for Y, 1 for U or V) */



    for (b=0; b<6; b++) {

        int ct = 0;    /* code type */

        if (b > 3) pt = 1;

        vlc_coeff = &s->dccv_vlc[pt];



        for (coeff_idx=0; coeff_idx<64; ) {

            int run = 1;

            if (coeff_idx<2 && s->nb_null[coeff_idx][pt]) {

                s->nb_null[coeff_idx][pt]--;

                if (coeff_idx)

                    break;

            } else {

                if (get_bits_count(&s->gb) >= s->gb.size_in_bits)

                    return;

                coeff = get_vlc2(&s->gb, vlc_coeff->table, 9, 3);

                if (coeff == 0) {

                    if (coeff_idx) {

                        int pt = (coeff_idx >= 6);

                        run += get_vlc2(&s->gb, s->runv_vlc[pt].table, 9, 3);

                        if (run >= 9)

                            run += get_bits(&s->gb, 6);

                    } else

                        s->nb_null[0][pt] = vp6_get_nb_null(s);

                    ct = 0;

                } else if (coeff == 11) {  /* end of block */

                    if (coeff_idx == 1)    /* first AC coeff ? */

                        s->nb_null[1][pt] = vp6_get_nb_null(s);

                    break;

                } else {

                    int coeff2 = vp56_coeff_bias[coeff];

                    if (coeff > 4)

                        coeff2 += get_bits(&s->gb, coeff <= 9 ? coeff - 4 : 11);

                    ct = 1 + (coeff2 > 1);

                    sign = get_bits1(&s->gb);

                    coeff2 = (coeff2 ^ -sign) + sign;

                    if (coeff_idx)

                        coeff2 *= s->dequant_ac;

                    idx = model->coeff_index_to_pos[coeff_idx];

                    s->block_coeff[b][permute[idx]] = coeff2;

                }

            }

            coeff_idx+=run;

            cg = FFMIN(vp6_coeff_groups[coeff_idx], 3);

            vlc_coeff = &s->ract_vlc[pt][ct][cg];

        }

    }

}
