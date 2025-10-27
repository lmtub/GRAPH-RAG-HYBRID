static int temporal_luma_motion_vector(HEVCContext *s, int x0, int y0,

                                       int nPbW, int nPbH, int refIdxLx,

                                       Mv *mvLXCol, int X)

{

    MvField *tab_mvf;

    MvField temp_col;

    int x, y, x_pu, y_pu;

    int min_pu_width = s->sps->min_pu_width;

    int availableFlagLXCol = 0;

    int colPic;



    HEVCFrame *ref = s->ref->collocated_ref;



    if (!ref)

        return 0;



    tab_mvf = ref->tab_mvf;

    colPic  = ref->poc;



    //bottom right collocated motion vector

    x = x0 + nPbW;

    y = y0 + nPbH;



    if (tab_mvf &&

        (y0 >> s->sps->log2_ctb_size) == (y >> s->sps->log2_ctb_size) &&

        y < s->sps->height &&

        x < s->sps->width) {

        x                 &= ~15;

        y                 &= ~15;

        ff_thread_await_progress(&ref->tf, y, 0);

        x_pu               = x >> s->sps->log2_min_pu_size;

        y_pu               = y >> s->sps->log2_min_pu_size;

        temp_col           = TAB_MVF(x_pu, y_pu);

        availableFlagLXCol = DERIVE_TEMPORAL_COLOCATED_MVS;

    }



    // derive center collocated motion vector

    if (tab_mvf && !availableFlagLXCol) {

        x                  = x0 + (nPbW >> 1);

        y                  = y0 + (nPbH >> 1);

        x                 &= ~15;

        y                 &= ~15;

        ff_thread_await_progress(&ref->tf, y, 0);

        x_pu               = x >> s->sps->log2_min_pu_size;

        y_pu               = y >> s->sps->log2_min_pu_size;

        temp_col           = TAB_MVF(x_pu, y_pu);

        availableFlagLXCol = DERIVE_TEMPORAL_COLOCATED_MVS;

    }

    return availableFlagLXCol;

}
