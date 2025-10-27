static void vp5_parse_coeff_models(VP56Context *s)

{

    VP56RangeCoder *c = &s->c;

    VP56Model *model = s->modelp;

    uint8_t def_prob[11];

    int node, cg, ctx;

    int ct;    /* code type */

    int pt;    /* plane type (0 for Y, 1 for U or V) */



    memset(def_prob, 0x80, sizeof(def_prob));



    for (pt=0; pt<2; pt++)

        for (node=0; node<11; node++)

            if (vp56_rac_get_prob(c, vp5_dccv_pct[pt][node])) {

                def_prob[node] = vp56_rac_gets_nn(c, 7);

                model->coeff_dccv[pt][node] = def_prob[node];

            } else if (s->framep[VP56_FRAME_CURRENT]->key_frame) {

                model->coeff_dccv[pt][node] = def_prob[node];

            }



    for (ct=0; ct<3; ct++)

        for (pt=0; pt<2; pt++)

            for (cg=0; cg<6; cg++)

                for (node=0; node<11; node++)

                    if (vp56_rac_get_prob(c, vp5_ract_pct[ct][pt][cg][node])) {

                        def_prob[node] = vp56_rac_gets_nn(c, 7);

                        model->coeff_ract[pt][ct][cg][node] = def_prob[node];

                    } else if (s->framep[VP56_FRAME_CURRENT]->key_frame) {

                        model->coeff_ract[pt][ct][cg][node] = def_prob[node];

                    }



    /* coeff_dcct is a linear combination of coeff_dccv */

    for (pt=0; pt<2; pt++)

        for (ctx=0; ctx<36; ctx++)

            for (node=0; node<5; node++)

                model->coeff_dcct[pt][ctx][node] = av_clip(((model->coeff_dccv[pt][node] * vp5_dccv_lc[node][ctx][0] + 128) >> 8) + vp5_dccv_lc[node][ctx][1], 1, 254);



    /* coeff_acct is a linear combination of coeff_ract */

    for (ct=0; ct<3; ct++)

        for (pt=0; pt<2; pt++)

            for (cg=0; cg<3; cg++)

                for (ctx=0; ctx<6; ctx++)

                    for (node=0; node<5; node++)

                        model->coeff_acct[pt][ct][cg][ctx][node] = av_clip(((model->coeff_ract[pt][ct][cg][node] * vp5_ract_lc[ct][cg][node][ctx][0] + 128) >> 8) + vp5_ract_lc[ct][cg][node][ctx][1], 1, 254);

}
