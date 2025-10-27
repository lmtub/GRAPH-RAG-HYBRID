static inline int direct_search(MpegEncContext * s, int mb_x, int mb_y)

{

    MotionEstContext * const c= &s->me;

    int P[10][2];

    const int mot_stride = s->mb_stride;

    const int mot_xy = mb_y*mot_stride + mb_x;

    const int shift= 1+s->quarter_sample;

    int dmin, i;

    const int time_pp= s->pp_time;

    const int time_pb= s->pb_time;

    int mx, my, xmin, xmax, ymin, ymax;

    int16_t (*mv_table)[2]= s->b_direct_mv_table;



    c->current_mv_penalty= c->mv_penalty[1] + MAX_MV;

    ymin= xmin=(-32)>>shift;

    ymax= xmax=   31>>shift;



    if (IS_8X8(s->next_picture.mb_type[mot_xy])) {

        s->mv_type= MV_TYPE_8X8;

    }else{

        s->mv_type= MV_TYPE_16X16;

    }



    for(i=0; i<4; i++){

        int index= s->block_index[i];

        int min, max;



        c->co_located_mv[i][0] = s->next_picture.motion_val[0][index][0];

        c->co_located_mv[i][1] = s->next_picture.motion_val[0][index][1];

        c->direct_basis_mv[i][0]= c->co_located_mv[i][0]*time_pb/time_pp + ((i& 1)<<(shift+3));

        c->direct_basis_mv[i][1]= c->co_located_mv[i][1]*time_pb/time_pp + ((i>>1)<<(shift+3));

//        c->direct_basis_mv[1][i][0]= c->co_located_mv[i][0]*(time_pb - time_pp)/time_pp + ((i &1)<<(shift+3);

//        c->direct_basis_mv[1][i][1]= c->co_located_mv[i][1]*(time_pb - time_pp)/time_pp + ((i>>1)<<(shift+3);



        max= FFMAX(c->direct_basis_mv[i][0], c->direct_basis_mv[i][0] - c->co_located_mv[i][0])>>shift;

        min= FFMIN(c->direct_basis_mv[i][0], c->direct_basis_mv[i][0] - c->co_located_mv[i][0])>>shift;

        max+= 16*mb_x + 1; // +-1 is for the simpler rounding

        min+= 16*mb_x - 1;

        xmax= FFMIN(xmax, s->width - max);

        xmin= FFMAX(xmin, - 16     - min);



        max= FFMAX(c->direct_basis_mv[i][1], c->direct_basis_mv[i][1] - c->co_located_mv[i][1])>>shift;

        min= FFMIN(c->direct_basis_mv[i][1], c->direct_basis_mv[i][1] - c->co_located_mv[i][1])>>shift;

        max+= 16*mb_y + 1; // +-1 is for the simpler rounding

        min+= 16*mb_y - 1;

        ymax= FFMIN(ymax, s->height - max);

        ymin= FFMAX(ymin, - 16      - min);



        if(s->mv_type == MV_TYPE_16X16) break;

    }



    av_assert2(xmax <= 15 && ymax <= 15 && xmin >= -16 && ymin >= -16);



    if(xmax < 0 || xmin >0 || ymax < 0 || ymin > 0){

        s->b_direct_mv_table[mot_xy][0]= 0;

        s->b_direct_mv_table[mot_xy][1]= 0;



        return 256*256*256*64;

    }



    c->xmin= xmin;

    c->ymin= ymin;

    c->xmax= xmax;

    c->ymax= ymax;

    c->flags     |= FLAG_DIRECT;

    c->sub_flags |= FLAG_DIRECT;

    c->pred_x=0;

    c->pred_y=0;



    P_LEFT[0]        = av_clip(mv_table[mot_xy - 1][0], xmin<<shift, xmax<<shift);

    P_LEFT[1]        = av_clip(mv_table[mot_xy - 1][1], ymin<<shift, ymax<<shift);



    /* special case for first line */

    if (!s->first_slice_line) { //FIXME maybe allow this over thread boundary as it is clipped

        P_TOP[0]      = av_clip(mv_table[mot_xy - mot_stride             ][0], xmin<<shift, xmax<<shift);

        P_TOP[1]      = av_clip(mv_table[mot_xy - mot_stride             ][1], ymin<<shift, ymax<<shift);

        P_TOPRIGHT[0] = av_clip(mv_table[mot_xy - mot_stride + 1         ][0], xmin<<shift, xmax<<shift);

        P_TOPRIGHT[1] = av_clip(mv_table[mot_xy - mot_stride + 1         ][1], ymin<<shift, ymax<<shift);



        P_MEDIAN[0]= mid_pred(P_LEFT[0], P_TOP[0], P_TOPRIGHT[0]);

        P_MEDIAN[1]= mid_pred(P_LEFT[1], P_TOP[1], P_TOPRIGHT[1]);

    }



    dmin = ff_epzs_motion_search(s, &mx, &my, P, 0, 0, mv_table, 1<<(16-shift), 0, 16);

    if(c->sub_flags&FLAG_QPEL)

        dmin = qpel_motion_search(s, &mx, &my, dmin, 0, 0, 0, 16);

    else

        dmin = hpel_motion_search(s, &mx, &my, dmin, 0, 0, 0, 16);



    if(c->avctx->me_sub_cmp != c->avctx->mb_cmp && !c->skip)

        dmin= get_mb_score(s, mx, my, 0, 0, 0, 16, 1);



    get_limits(s, 16*mb_x, 16*mb_y); //restore c->?min/max, maybe not needed



    mv_table[mot_xy][0]= mx;

    mv_table[mot_xy][1]= my;

    c->flags     &= ~FLAG_DIRECT;

    c->sub_flags &= ~FLAG_DIRECT;



    return dmin;

}
