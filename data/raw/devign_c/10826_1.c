int ff_pre_estimate_p_frame_motion(MpegEncContext * s,

                                    int mb_x, int mb_y)

{

    MotionEstContext * const c= &s->me;

    int mx, my, dmin;

    int P[10][2];

    const int shift= 1+s->quarter_sample;

    const int xy= mb_x + mb_y*s->mb_stride;

    init_ref(c, s->new_picture.f.data, s->last_picture.f.data, NULL, 16*mb_x, 16*mb_y, 0);



    assert(s->quarter_sample==0 || s->quarter_sample==1);



    c->pre_penalty_factor    = get_penalty_factor(s->lambda, s->lambda2, c->avctx->me_pre_cmp);

    c->current_mv_penalty= c->mv_penalty[s->f_code] + MAX_MV;



    get_limits(s, 16*mb_x, 16*mb_y);

    c->skip=0;



    P_LEFT[0]       = s->p_mv_table[xy + 1][0];

    P_LEFT[1]       = s->p_mv_table[xy + 1][1];



    if(P_LEFT[0]       < (c->xmin<<shift)) P_LEFT[0]       = (c->xmin<<shift);



    /* special case for first line */

    if (s->first_slice_line) {

        c->pred_x= P_LEFT[0];

        c->pred_y= P_LEFT[1];

        P_TOP[0]= P_TOPRIGHT[0]= P_MEDIAN[0]=

        P_TOP[1]= P_TOPRIGHT[1]= P_MEDIAN[1]= 0; //FIXME

    } else {

        P_TOP[0]      = s->p_mv_table[xy + s->mb_stride    ][0];

        P_TOP[1]      = s->p_mv_table[xy + s->mb_stride    ][1];

        P_TOPRIGHT[0] = s->p_mv_table[xy + s->mb_stride - 1][0];

        P_TOPRIGHT[1] = s->p_mv_table[xy + s->mb_stride - 1][1];

        if(P_TOP[1]      < (c->ymin<<shift)) P_TOP[1]     = (c->ymin<<shift);

        if(P_TOPRIGHT[0] > (c->xmax<<shift)) P_TOPRIGHT[0]= (c->xmax<<shift);

        if(P_TOPRIGHT[1] < (c->ymin<<shift)) P_TOPRIGHT[1]= (c->ymin<<shift);



        P_MEDIAN[0]= mid_pred(P_LEFT[0], P_TOP[0], P_TOPRIGHT[0]);

        P_MEDIAN[1]= mid_pred(P_LEFT[1], P_TOP[1], P_TOPRIGHT[1]);



        c->pred_x = P_MEDIAN[0];

        c->pred_y = P_MEDIAN[1];

    }



    dmin = ff_epzs_motion_search(s, &mx, &my, P, 0, 0, s->p_mv_table, (1<<16)>>shift, 0, 16);



    s->p_mv_table[xy][0] = mx<<shift;

    s->p_mv_table[xy][1] = my<<shift;



    return dmin;

}
