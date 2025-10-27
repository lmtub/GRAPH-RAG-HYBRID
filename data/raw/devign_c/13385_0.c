void decode_mvs(VP8Context *s, VP8Macroblock *mb, int mb_x, int mb_y)

{

    VP8Macroblock *mb_edge[3] = { mb + 2 /* top */,

                                  mb - 1 /* left */,

                                  mb + 1 /* top-left */ };

    enum { CNT_ZERO, CNT_NEAREST, CNT_NEAR, CNT_SPLITMV };

    enum { EDGE_TOP, EDGE_LEFT, EDGE_TOPLEFT };

    int idx = CNT_ZERO;

    int cur_sign_bias = s->sign_bias[mb->ref_frame];

    int *sign_bias = s->sign_bias;

    VP56mv near_mv[4];

    uint8_t cnt[4] = { 0 };

    VP56RangeCoder *c = &s->c;



    AV_ZERO32(&near_mv[0]);

    AV_ZERO32(&near_mv[1]);

    AV_ZERO32(&near_mv[2]);



    /* Process MB on top, left and top-left */

    #define MV_EDGE_CHECK(n)\

    {\

        VP8Macroblock *edge = mb_edge[n];\

        int edge_ref = edge->ref_frame;\

        if (edge_ref != VP56_FRAME_CURRENT) {\

            uint32_t mv = AV_RN32A(&edge->mv);\

            if (mv) {\

                if (cur_sign_bias != sign_bias[edge_ref]) {\

                    /* SWAR negate of the values in mv. */\

                    mv = ~mv;\

                    mv = ((mv&0x7fff7fff) + 0x00010001) ^ (mv&0x80008000);\

                }\

                if (!n || mv != AV_RN32A(&near_mv[idx]))\

                    AV_WN32A(&near_mv[++idx], mv);\

                cnt[idx]      += 1 + (n != 2);\

            } else\

                cnt[CNT_ZERO] += 1 + (n != 2);\

        }\

    }



    MV_EDGE_CHECK(0)

    MV_EDGE_CHECK(1)

    MV_EDGE_CHECK(2)



    mb->partitioning = VP8_SPLITMVMODE_NONE;

    if (vp56_rac_get_prob_branchy(c, vp8_mode_contexts[cnt[CNT_ZERO]][0])) {

        mb->mode = VP8_MVMODE_MV;



        /* If we have three distinct MVs, merge first and last if they're the same */

        if (cnt[CNT_SPLITMV] && AV_RN32A(&near_mv[1+EDGE_TOP]) == AV_RN32A(&near_mv[1+EDGE_TOPLEFT]))

            cnt[CNT_NEAREST] += 1;



        /* Swap near and nearest if necessary */

        if (cnt[CNT_NEAR] > cnt[CNT_NEAREST]) {

            FFSWAP(uint8_t,     cnt[CNT_NEAREST],     cnt[CNT_NEAR]);

            FFSWAP( VP56mv, near_mv[CNT_NEAREST], near_mv[CNT_NEAR]);

        }



        if (vp56_rac_get_prob_branchy(c, vp8_mode_contexts[cnt[CNT_NEAREST]][1])) {

            if (vp56_rac_get_prob_branchy(c, vp8_mode_contexts[cnt[CNT_NEAR]][2])) {



                /* Choose the best mv out of 0,0 and the nearest mv */

                clamp_mv(s, &mb->mv, &near_mv[CNT_ZERO + (cnt[CNT_NEAREST] >= cnt[CNT_ZERO])]);

                cnt[CNT_SPLITMV] = ((mb_edge[EDGE_LEFT]->mode    == VP8_MVMODE_SPLIT) +

                                    (mb_edge[EDGE_TOP]->mode     == VP8_MVMODE_SPLIT)) * 2 +

                                    (mb_edge[EDGE_TOPLEFT]->mode == VP8_MVMODE_SPLIT);



                if (vp56_rac_get_prob_branchy(c, vp8_mode_contexts[cnt[CNT_SPLITMV]][3])) {

                    mb->mode = VP8_MVMODE_SPLIT;

                    mb->mv = mb->bmv[decode_splitmvs(s, c, mb) - 1];

                } else {

                    mb->mv.y += read_mv_component(c, s->prob->mvc[0]);

                    mb->mv.x += read_mv_component(c, s->prob->mvc[1]);

                    mb->bmv[0] = mb->mv;

                }

            } else {

                clamp_mv(s, &mb->mv, &near_mv[CNT_NEAR]);

                mb->bmv[0] = mb->mv;

            }

        } else {

            clamp_mv(s, &mb->mv, &near_mv[CNT_NEAREST]);

            mb->bmv[0] = mb->mv;

        }

    } else {

        mb->mode = VP8_MVMODE_ZERO;

        AV_ZERO32(&mb->mv);

        mb->bmv[0] = mb->mv;

    }

}
