static int rv34_decode_mv(RV34DecContext *r, int block_type)

{

    MpegEncContext *s = &r->s;

    GetBitContext *gb = &s->gb;

    int i, j, k, l;

    int mv_pos = s->mb_x * 2 + s->mb_y * 2 * s->b8_stride;

    int next_bt;



    memset(r->dmv, 0, sizeof(r->dmv));

    for(i = 0; i < num_mvs[block_type]; i++){

        r->dmv[i][0] = svq3_get_se_golomb(gb);

        r->dmv[i][1] = svq3_get_se_golomb(gb);

    }

    switch(block_type){

    case RV34_MB_TYPE_INTRA:

    case RV34_MB_TYPE_INTRA16x16:

        ZERO8x2(s->current_picture_ptr->f.motion_val[0][s->mb_x * 2 + s->mb_y * 2 * s->b8_stride], s->b8_stride);

        return 0;

    case RV34_MB_SKIP:

        if(s->pict_type == AV_PICTURE_TYPE_P){

            ZERO8x2(s->current_picture_ptr->f.motion_val[0][s->mb_x * 2 + s->mb_y * 2 * s->b8_stride], s->b8_stride);

            rv34_mc_1mv (r, block_type, 0, 0, 0, 2, 2, 0);

            break;

        }

    case RV34_MB_B_DIRECT:

        //surprisingly, it uses motion scheme from next reference frame

        /* wait for the current mb row to be finished */

        if (HAVE_THREADS && (s->avctx->active_thread_type & FF_THREAD_FRAME))

            ff_thread_await_progress(&s->next_picture_ptr->f, s->mb_y - 1, 0);



        next_bt = s->next_picture_ptr->f.mb_type[s->mb_x + s->mb_y * s->mb_stride];

        if(IS_INTRA(next_bt) || IS_SKIP(next_bt)){

            ZERO8x2(s->current_picture_ptr->f.motion_val[0][s->mb_x * 2 + s->mb_y * 2 * s->b8_stride], s->b8_stride);

            ZERO8x2(s->current_picture_ptr->f.motion_val[1][s->mb_x * 2 + s->mb_y * 2 * s->b8_stride], s->b8_stride);

        }else

            for(j = 0; j < 2; j++)

                for(i = 0; i < 2; i++)

                    for(k = 0; k < 2; k++)

                        for(l = 0; l < 2; l++)

                            s->current_picture_ptr->f.motion_val[l][mv_pos + i + j*s->b8_stride][k] = calc_add_mv(r, l, s->next_picture_ptr->f.motion_val[0][mv_pos + i + j*s->b8_stride][k]);

        if(!(IS_16X8(next_bt) || IS_8X16(next_bt) || IS_8X8(next_bt))) //we can use whole macroblock MC

            rv34_mc_2mv(r, block_type);

        else

            rv34_mc_2mv_skip(r);

        ZERO8x2(s->current_picture_ptr->f.motion_val[0][s->mb_x * 2 + s->mb_y * 2 * s->b8_stride], s->b8_stride);

        break;

    case RV34_MB_P_16x16:

    case RV34_MB_P_MIX16x16:

        rv34_pred_mv(r, block_type, 0, 0);

        rv34_mc_1mv (r, block_type, 0, 0, 0, 2, 2, 0);

        break;

    case RV34_MB_B_FORWARD:

    case RV34_MB_B_BACKWARD:

        r->dmv[1][0] = r->dmv[0][0];

        r->dmv[1][1] = r->dmv[0][1];

        if(r->rv30)

            rv34_pred_mv_rv3(r, block_type, block_type == RV34_MB_B_BACKWARD);

        else

            rv34_pred_mv_b  (r, block_type, block_type == RV34_MB_B_BACKWARD);

        rv34_mc_1mv     (r, block_type, 0, 0, 0, 2, 2, block_type == RV34_MB_B_BACKWARD);

        break;

    case RV34_MB_P_16x8:

    case RV34_MB_P_8x16:

        rv34_pred_mv(r, block_type, 0, 0);

        rv34_pred_mv(r, block_type, 1 + (block_type == RV34_MB_P_16x8), 1);

        if(block_type == RV34_MB_P_16x8){

            rv34_mc_1mv(r, block_type, 0, 0, 0,            2, 1, 0);

            rv34_mc_1mv(r, block_type, 0, 8, s->b8_stride, 2, 1, 0);

        }

        if(block_type == RV34_MB_P_8x16){

            rv34_mc_1mv(r, block_type, 0, 0, 0, 1, 2, 0);

            rv34_mc_1mv(r, block_type, 8, 0, 1, 1, 2, 0);

        }

        break;

    case RV34_MB_B_BIDIR:

        rv34_pred_mv_b  (r, block_type, 0);

        rv34_pred_mv_b  (r, block_type, 1);

        rv34_mc_2mv     (r, block_type);

        break;

    case RV34_MB_P_8x8:

        for(i=0;i< 4;i++){

            rv34_pred_mv(r, block_type, i, i);

            rv34_mc_1mv (r, block_type, (i&1)<<3, (i&2)<<2, (i&1)+(i>>1)*s->b8_stride, 1, 1, 0);

        }

        break;

    }



    return 0;

}
