static void avc_loopfilter_luma_intra_edge_hor_msa(uint8_t *data,

                                                   uint8_t alpha_in,

                                                   uint8_t beta_in,

                                                   uint32_t img_width)

{

    v16u8 p2_asub_p0, q2_asub_q0, p0_asub_q0;

    v16u8 alpha, beta;

    v16u8 is_less_than, is_less_than_beta, negate_is_less_than_beta;

    v16u8 p2, p1, p0, q0, q1, q2;

    v16u8 p3_org, p2_org, p1_org, p0_org, q0_org, q1_org, q2_org, q3_org;

    v8i16 p1_org_r, p0_org_r, q0_org_r, q1_org_r;

    v8i16 p1_org_l, p0_org_l, q0_org_l, q1_org_l;

    v8i16 p2_r = { 0 };

    v8i16 p1_r = { 0 };

    v8i16 p0_r = { 0 };

    v8i16 q0_r = { 0 };

    v8i16 q1_r = { 0 };

    v8i16 q2_r = { 0 };

    v8i16 p2_l = { 0 };

    v8i16 p1_l = { 0 };

    v8i16 p0_l = { 0 };

    v8i16 q0_l = { 0 };

    v8i16 q1_l = { 0 };

    v8i16 q2_l = { 0 };

    v16u8 tmp_flag;

    v16i8 zero = { 0 };



    alpha = (v16u8) __msa_fill_b(alpha_in);

    beta = (v16u8) __msa_fill_b(beta_in);



    p1_org = LOAD_UB(data - (img_width << 1));

    p0_org = LOAD_UB(data - img_width);

    q0_org = LOAD_UB(data);

    q1_org = LOAD_UB(data + img_width);



    {

        v16u8 p1_asub_p0, q1_asub_q0, is_less_than_alpha;



        p0_asub_q0 = __msa_asub_u_b(p0_org, q0_org);

        p1_asub_p0 = __msa_asub_u_b(p1_org, p0_org);

        q1_asub_q0 = __msa_asub_u_b(q1_org, q0_org);



        is_less_than_alpha = (p0_asub_q0 < alpha);

        is_less_than_beta = (p1_asub_p0 < beta);

        is_less_than = is_less_than_beta & is_less_than_alpha;

        is_less_than_beta = (q1_asub_q0 < beta);

        is_less_than = is_less_than_beta & is_less_than;

    }



    if (!__msa_test_bz_v(is_less_than)) {

        q2_org = LOAD_UB(data + (2 * img_width));

        p3_org = LOAD_UB(data - (img_width << 2));

        p2_org = LOAD_UB(data - (3 * img_width));



        p1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p1_org);

        p0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p0_org);

        q0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q0_org);



        p1_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p1_org);

        p0_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p0_org);

        q0_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q0_org);



        tmp_flag = alpha >> 2;

        tmp_flag = tmp_flag + 2;

        tmp_flag = (p0_asub_q0 < tmp_flag);



        p2_asub_p0 = __msa_asub_u_b(p2_org, p0_org);

        is_less_than_beta = (p2_asub_p0 < beta);

        is_less_than_beta = is_less_than_beta & tmp_flag;



        negate_is_less_than_beta = __msa_xori_b(is_less_than_beta, 0xff);

        is_less_than_beta = is_less_than_beta & is_less_than;

        negate_is_less_than_beta = negate_is_less_than_beta & is_less_than;



        {

            v8u16 is_less_than_beta_l, is_less_than_beta_r;



            q1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q1_org);



            is_less_than_beta_r =

                (v8u16) __msa_sldi_b((v16i8) is_less_than_beta, zero, 8);

            if (!__msa_test_bz_v((v16u8) is_less_than_beta_r)) {

                v8i16 p3_org_r;



                p3_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p3_org);

                p2_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p2_org);



                AVC_LOOP_FILTER_P0P1P2_OR_Q0Q1Q2(p3_org_r, p0_org_r,

                                                 q0_org_r, p1_org_r,

                                                 p2_r, q1_org_r,

                                                 p0_r, p1_r, p2_r);

            }



            q1_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q1_org);



            is_less_than_beta_l =

                (v8u16) __msa_sldi_b(zero, (v16i8) is_less_than_beta, 8);



            if (!__msa_test_bz_v((v16u8) is_less_than_beta_l)) {

                v8i16 p3_org_l;



                p3_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p3_org);

                p2_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p2_org);



                AVC_LOOP_FILTER_P0P1P2_OR_Q0Q1Q2(p3_org_l, p0_org_l,

                                                 q0_org_l, p1_org_l,

                                                 p2_l, q1_org_l,

                                                 p0_l, p1_l, p2_l);

            }

        }



        /* combine and store */

        if (!__msa_test_bz_v(is_less_than_beta)) {

            p0 = (v16u8) __msa_pckev_b((v16i8) p0_l, (v16i8) p0_r);

            p1 = (v16u8) __msa_pckev_b((v16i8) p1_l, (v16i8) p1_r);

            p2 = (v16u8) __msa_pckev_b((v16i8) p2_l, (v16i8) p2_r);



            p0_org = __msa_bmnz_v(p0_org, p0, is_less_than_beta);

            p1_org = __msa_bmnz_v(p1_org, p1, is_less_than_beta);

            p2_org = __msa_bmnz_v(p2_org, p2, is_less_than_beta);



            STORE_UB(p1_org, data - (2 * img_width));

            STORE_UB(p2_org, data - (3 * img_width));

        }



        {

            v8u16 negate_is_less_than_beta_r, negate_is_less_than_beta_l;



            negate_is_less_than_beta_r =

                (v8u16) __msa_sldi_b((v16i8) negate_is_less_than_beta, zero, 8);

            if (!__msa_test_bz_v((v16u8) negate_is_less_than_beta_r)) {

                AVC_LOOP_FILTER_P0_OR_Q0(p0_org_r, q1_org_r, p1_org_r, p0_r);

            }



            negate_is_less_than_beta_l =

                (v8u16) __msa_sldi_b(zero, (v16i8) negate_is_less_than_beta, 8);

            if (!__msa_test_bz_v((v16u8) negate_is_less_than_beta_l)) {

                AVC_LOOP_FILTER_P0_OR_Q0(p0_org_l, q1_org_l, p1_org_l, p0_l);

            }

        }



        /* combine */

        if (!__msa_test_bz_v(negate_is_less_than_beta)) {

            p0 = (v16u8) __msa_pckev_b((v16i8) p0_l, (v16i8) p0_r);

            p0_org = __msa_bmnz_v(p0_org, p0, negate_is_less_than_beta);

        }



        STORE_UB(p0_org, data - img_width);



        /* if (tmpFlag && (unsigned)ABS(q2-q0) < thresholds->beta_in) */



        q3_org = LOAD_UB(data + (3 * img_width));



        q2_asub_q0 = __msa_asub_u_b(q2_org, q0_org);

        is_less_than_beta = (q2_asub_q0 < beta);

        is_less_than_beta = is_less_than_beta & tmp_flag;

        negate_is_less_than_beta = __msa_xori_b(is_less_than_beta, 0xff);

        is_less_than_beta = is_less_than_beta & is_less_than;

        negate_is_less_than_beta = negate_is_less_than_beta & is_less_than;



        {

            v8u16 is_less_than_beta_l, is_less_than_beta_r;



            is_less_than_beta_r =

                (v8u16) __msa_sldi_b((v16i8) is_less_than_beta, zero, 8);

            if (!__msa_test_bz_v((v16u8) is_less_than_beta_r)) {

                v8i16 q3_org_r;



                q3_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q3_org);

                q2_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q2_org);



                AVC_LOOP_FILTER_P0P1P2_OR_Q0Q1Q2(q3_org_r, q0_org_r,

                                                 p0_org_r, q1_org_r,

                                                 q2_r, p1_org_r,

                                                 q0_r, q1_r, q2_r);

            }



            is_less_than_beta_l =

                (v8u16) __msa_sldi_b(zero, (v16i8) is_less_than_beta, 8);

            if (!__msa_test_bz_v((v16u8) is_less_than_beta_l)) {

                v8i16 q3_org_l;



                q3_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q3_org);

                q2_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q2_org);



                AVC_LOOP_FILTER_P0P1P2_OR_Q0Q1Q2(q3_org_l, q0_org_l,

                                                 p0_org_l, q1_org_l,

                                                 q2_l, p1_org_l,

                                                 q0_l, q1_l, q2_l);

            }

        }



        /* combine and store */

        if (!__msa_test_bz_v(is_less_than_beta)) {

            q0 = (v16u8) __msa_pckev_b((v16i8) q0_l, (v16i8) q0_r);

            q1 = (v16u8) __msa_pckev_b((v16i8) q1_l, (v16i8) q1_r);

            q2 = (v16u8) __msa_pckev_b((v16i8) q2_l, (v16i8) q2_r);



            q0_org = __msa_bmnz_v(q0_org, q0, is_less_than_beta);

            q1_org = __msa_bmnz_v(q1_org, q1, is_less_than_beta);

            q2_org = __msa_bmnz_v(q2_org, q2, is_less_than_beta);



            STORE_UB(q1_org, data + img_width);

            STORE_UB(q2_org, data + 2 * img_width);

        }



        {

            v8u16 negate_is_less_than_beta_r, negate_is_less_than_beta_l;



            negate_is_less_than_beta_r =

                (v8u16) __msa_sldi_b((v16i8) negate_is_less_than_beta, zero, 8);

            if (!__msa_test_bz_v((v16u8) negate_is_less_than_beta_r)) {

                AVC_LOOP_FILTER_P0_OR_Q0(q0_org_r, p1_org_r, q1_org_r, q0_r);

            }



            negate_is_less_than_beta_l =

                (v8u16) __msa_sldi_b(zero, (v16i8) negate_is_less_than_beta, 8);

            if (!__msa_test_bz_v((v16u8) negate_is_less_than_beta_l)) {

                AVC_LOOP_FILTER_P0_OR_Q0(q0_org_l, p1_org_l, q1_org_l, q0_l);

            }

        }



        /* combine */

        if (!__msa_test_bz_v(negate_is_less_than_beta)) {

            q0 = (v16u8) __msa_pckev_b((v16i8) q0_l, (v16i8) q0_r);

            q0_org = __msa_bmnz_v(q0_org, q0, negate_is_less_than_beta);

        }



        STORE_UB(q0_org, data);

    }

}
