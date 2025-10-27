static void derive_spatial_merge_candidates(HEVCContext *s, int x0, int y0,

                                            int nPbW, int nPbH,

                                            int log2_cb_size,

                                            int singleMCLFlag, int part_idx,

                                            int merge_idx,

                                            struct MvField mergecandlist[])

{

    HEVCLocalContext *lc   = &s->HEVClc;

    RefPicList *refPicList = s->ref->refPicList;

    MvField *tab_mvf       = s->ref->tab_mvf;



    const int min_pu_width = s->sps->min_pu_width;



    const int cand_bottom_left = lc->na.cand_bottom_left;

    const int cand_left        = lc->na.cand_left;

    const int cand_up_left     = lc->na.cand_up_left;

    const int cand_up          = lc->na.cand_up;

    const int cand_up_right    = lc->na.cand_up_right_sap;



    const int xA1    = x0 - 1;

    const int yA1    = y0 + nPbH - 1;

    const int xA1_pu = xA1 >> s->sps->log2_min_pu_size;

    const int yA1_pu = yA1 >> s->sps->log2_min_pu_size;



    const int xB1    = x0 + nPbW - 1;

    const int yB1    = y0 - 1;

    const int xB1_pu = xB1 >> s->sps->log2_min_pu_size;

    const int yB1_pu = yB1 >> s->sps->log2_min_pu_size;



    const int xB0    = x0 + nPbW;

    const int yB0    = y0 - 1;

    const int xB0_pu = xB0 >> s->sps->log2_min_pu_size;

    const int yB0_pu = yB0 >> s->sps->log2_min_pu_size;



    const int xA0    = x0 - 1;

    const int yA0    = y0 + nPbH;

    const int xA0_pu = xA0 >> s->sps->log2_min_pu_size;

    const int yA0_pu = yA0 >> s->sps->log2_min_pu_size;



    const int xB2    = x0 - 1;

    const int yB2    = y0 - 1;

    const int xB2_pu = xB2 >> s->sps->log2_min_pu_size;

    const int yB2_pu = yB2 >> s->sps->log2_min_pu_size;



    const int nb_refs = (s->sh.slice_type == P_SLICE) ?

                        s->sh.nb_refs[0] : FFMIN(s->sh.nb_refs[0], s->sh.nb_refs[1]);

    int check_MER   = 1;

    int check_MER_1 = 1;



    int zero_idx = 0;



    int nb_merge_cand = 0;

    int nb_orig_merge_cand = 0;



    int is_available_a0;

    int is_available_a1;

    int is_available_b0;

    int is_available_b1;

    int is_available_b2;

    int check_B0;

    int check_A0;



    //first left spatial merge candidate

    is_available_a1 = AVAILABLE(cand_left, A1);



    if (!singleMCLFlag && part_idx == 1 &&

        (lc->cu.part_mode == PART_Nx2N ||

         lc->cu.part_mode == PART_nLx2N ||

         lc->cu.part_mode == PART_nRx2N) ||

        isDiffMER(s, xA1, yA1, x0, y0)) {

        is_available_a1 = 0;

    }



    if (is_available_a1) {

        mergecandlist[0] = TAB_MVF_PU(A1);

        if (merge_idx == 0)

            return;

        nb_merge_cand++;

    }



    // above spatial merge candidate

    is_available_b1 = AVAILABLE(cand_up, B1);



    if (!singleMCLFlag && part_idx == 1 &&

        (lc->cu.part_mode == PART_2NxN ||

         lc->cu.part_mode == PART_2NxnU ||

         lc->cu.part_mode == PART_2NxnD) ||

        isDiffMER(s, xB1, yB1, x0, y0)) {

        is_available_b1 = 0;

    }



    if (is_available_a1 && is_available_b1)

        check_MER = !COMPARE_MV_REFIDX(B1, A1);



    if (is_available_b1 && check_MER)

        mergecandlist[nb_merge_cand++] = TAB_MVF_PU(B1);



    // above right spatial merge candidate

    check_MER = 1;

    check_B0  = PRED_BLOCK_AVAILABLE(B0);



    is_available_b0 = check_B0 && AVAILABLE(cand_up_right, B0);



    if (isDiffMER(s, xB0, yB0, x0, y0))

        is_available_b0 = 0;



    if (is_available_b1 && is_available_b0)

        check_MER = !COMPARE_MV_REFIDX(B0, B1);



    if (is_available_b0 && check_MER) {

        mergecandlist[nb_merge_cand] = TAB_MVF_PU(B0);

        if (merge_idx == nb_merge_cand)

            return;

        nb_merge_cand++;

    }



    // left bottom spatial merge candidate

    check_MER = 1;

    check_A0  = PRED_BLOCK_AVAILABLE(A0);



    is_available_a0 = check_A0 && AVAILABLE(cand_bottom_left, A0);



    if (isDiffMER(s, xA0, yA0, x0, y0))

        is_available_a0 = 0;



    if (is_available_a1 && is_available_a0)

        check_MER = !COMPARE_MV_REFIDX(A0, A1);



    if (is_available_a0 && check_MER) {

        mergecandlist[nb_merge_cand] = TAB_MVF_PU(A0);

        if (merge_idx == nb_merge_cand)

            return;

        nb_merge_cand++;

    }



    // above left spatial merge candidate

    check_MER = 1;



    is_available_b2 = AVAILABLE(cand_up_left, B2);



    if (isDiffMER(s, xB2, yB2, x0, y0))

        is_available_b2 = 0;



    if (is_available_a1 && is_available_b2)

        check_MER = !COMPARE_MV_REFIDX(B2, A1);



    if (is_available_b1 && is_available_b2)

        check_MER_1 = !COMPARE_MV_REFIDX(B2, B1);



    if (is_available_b2 && check_MER && check_MER_1 && nb_merge_cand != 4) {

        mergecandlist[nb_merge_cand] = TAB_MVF_PU(B2);

        if (merge_idx == nb_merge_cand)

            return;

        nb_merge_cand++;

    }



    // temporal motion vector candidate

    if (s->sh.slice_temporal_mvp_enabled_flag &&

        nb_merge_cand < s->sh.max_num_merge_cand) {

        Mv mv_l0_col, mv_l1_col;

        int available_l0 = temporal_luma_motion_vector(s, x0, y0, nPbW, nPbH,

                                                       0, &mv_l0_col, 0);

        int available_l1 = (s->sh.slice_type == B_SLICE) ?

                           temporal_luma_motion_vector(s, x0, y0, nPbW, nPbH,

                                                       0, &mv_l1_col, 1) : 0;



        if (available_l0 || available_l1) {

            mergecandlist[nb_merge_cand].is_intra     = 0;

            mergecandlist[nb_merge_cand].pred_flag[0] = available_l0;

            mergecandlist[nb_merge_cand].pred_flag[1] = available_l1;

            if (available_l0) {

                mergecandlist[nb_merge_cand].mv[0]      = mv_l0_col;

                mergecandlist[nb_merge_cand].ref_idx[0] = 0;

            }

            if (available_l1) {

                mergecandlist[nb_merge_cand].mv[1]      = mv_l1_col;

                mergecandlist[nb_merge_cand].ref_idx[1] = 0;

            }

            if (merge_idx == nb_merge_cand)

                return;

            nb_merge_cand++;

        }

    }



    nb_orig_merge_cand = nb_merge_cand;



    // combined bi-predictive merge candidates  (applies for B slices)

    if (s->sh.slice_type == B_SLICE && nb_orig_merge_cand > 1 &&

        nb_orig_merge_cand < s->sh.max_num_merge_cand) {

        int comb_idx;



        for (comb_idx = 0; nb_merge_cand < s->sh.max_num_merge_cand &&

                           comb_idx < nb_orig_merge_cand * (nb_orig_merge_cand - 1); comb_idx++) {

            int l0_cand_idx = l0_l1_cand_idx[comb_idx][0];

            int l1_cand_idx = l0_l1_cand_idx[comb_idx][1];

            MvField l0_cand = mergecandlist[l0_cand_idx];

            MvField l1_cand = mergecandlist[l1_cand_idx];



            if (l0_cand.pred_flag[0] && l1_cand.pred_flag[1] &&

                (refPicList[0].list[l0_cand.ref_idx[0]] !=

                 refPicList[1].list[l1_cand.ref_idx[1]] ||

                 AV_RN32A(&l0_cand.mv[0]) != AV_RN32A(&l1_cand.mv[1]))) {

                mergecandlist[nb_merge_cand].ref_idx[0]   = l0_cand.ref_idx[0];

                mergecandlist[nb_merge_cand].ref_idx[1]   = l1_cand.ref_idx[1];

                mergecandlist[nb_merge_cand].pred_flag[0] = 1;

                mergecandlist[nb_merge_cand].pred_flag[1] = 1;

                AV_COPY32(&mergecandlist[nb_merge_cand].mv[0], &l0_cand.mv[0]);

                AV_COPY32(&mergecandlist[nb_merge_cand].mv[1], &l1_cand.mv[1]);

                mergecandlist[nb_merge_cand].is_intra     = 0;

                if (merge_idx == nb_merge_cand)

                    return;

                nb_merge_cand++;

            }

        }

    }



    // append Zero motion vector candidates

    while (nb_merge_cand < s->sh.max_num_merge_cand) {

        mergecandlist[nb_merge_cand].pred_flag[0] = 1;

        mergecandlist[nb_merge_cand].pred_flag[1] = s->sh.slice_type == B_SLICE;

        AV_ZERO32(mergecandlist[nb_merge_cand].mv + 0);

        AV_ZERO32(mergecandlist[nb_merge_cand].mv + 1);

        mergecandlist[nb_merge_cand].is_intra     = 0;

        mergecandlist[nb_merge_cand].ref_idx[0]   = zero_idx < nb_refs ? zero_idx : 0;

        mergecandlist[nb_merge_cand].ref_idx[1]   = zero_idx < nb_refs ? zero_idx : 0;



        if (merge_idx == nb_merge_cand)

            return;

        nb_merge_cand++;

        zero_idx++;

    }

}
