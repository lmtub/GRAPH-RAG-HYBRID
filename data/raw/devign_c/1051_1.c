void ff_estimate_b_frame_motion(MpegEncContext * s,

                             int mb_x, int mb_y)

{

    MotionEstContext * const c= &s->me;

    const int penalty_factor= c->mb_penalty_factor;

    int fmin, bmin, dmin, fbmin, bimin, fimin;

    int type=0;

    const int xy = mb_y*s->mb_stride + mb_x;

    init_ref(c, s->new_picture.f.data, s->last_picture.f.data,

             s->next_picture.f.data, 16 * mb_x, 16 * mb_y, 2);



    get_limits(s, 16*mb_x, 16*mb_y);



    c->skip=0;



    if (s->codec_id == AV_CODEC_ID_MPEG4 && s->next_picture.mbskip_table[xy]) {

        int score= direct_search(s, mb_x, mb_y); //FIXME just check 0,0



        score= ((unsigned)(score*score + 128*256))>>16;

        c->mc_mb_var_sum_temp += score;

        s->current_picture.mc_mb_var[mb_y*s->mb_stride + mb_x] = score; //FIXME use SSE

        s->mb_type[mb_y*s->mb_stride + mb_x]= CANDIDATE_MB_TYPE_DIRECT0;



        return;

    }



    if (s->codec_id == AV_CODEC_ID_MPEG4)

        dmin= direct_search(s, mb_x, mb_y);

    else

        dmin= INT_MAX;

//FIXME penalty stuff for non mpeg4

    c->skip=0;

    fmin = estimate_motion_b(s, mb_x, mb_y, s->b_forw_mv_table, 0, s->f_code) +

           3 * penalty_factor;



    c->skip=0;

    bmin = estimate_motion_b(s, mb_x, mb_y, s->b_back_mv_table, 2, s->b_code) +

           2 * penalty_factor;

    av_dlog(s, " %d %d ", s->b_forw_mv_table[xy][0], s->b_forw_mv_table[xy][1]);



    c->skip=0;

    fbmin= bidir_refine(s, mb_x, mb_y) + penalty_factor;

    av_dlog(s, "%d %d %d %d\n", dmin, fmin, bmin, fbmin);



    if(s->flags & CODEC_FLAG_INTERLACED_ME){

//FIXME mb type penalty

        c->skip=0;

        c->current_mv_penalty= c->mv_penalty[s->f_code] + MAX_MV;

        fimin= interlaced_search(s, 0,

                                 s->b_field_mv_table[0], s->b_field_select_table[0],

                                 s->b_forw_mv_table[xy][0], s->b_forw_mv_table[xy][1], 0);

        c->current_mv_penalty= c->mv_penalty[s->b_code] + MAX_MV;

        bimin= interlaced_search(s, 2,

                                 s->b_field_mv_table[1], s->b_field_select_table[1],

                                 s->b_back_mv_table[xy][0], s->b_back_mv_table[xy][1], 0);

    }else

        fimin= bimin= INT_MAX;



    {

        int score= fmin;

        type = CANDIDATE_MB_TYPE_FORWARD;



        if (dmin <= score){

            score = dmin;

            type = CANDIDATE_MB_TYPE_DIRECT;

        }

        if(bmin<score){

            score=bmin;

            type= CANDIDATE_MB_TYPE_BACKWARD;

        }

        if(fbmin<score){

            score=fbmin;

            type= CANDIDATE_MB_TYPE_BIDIR;

        }

        if(fimin<score){

            score=fimin;

            type= CANDIDATE_MB_TYPE_FORWARD_I;

        }

        if(bimin<score){

            score=bimin;

            type= CANDIDATE_MB_TYPE_BACKWARD_I;

        }



        score= ((unsigned)(score*score + 128*256))>>16;

        c->mc_mb_var_sum_temp += score;

        s->current_picture.mc_mb_var[mb_y*s->mb_stride + mb_x] = score; //FIXME use SSE

    }



    if(c->avctx->mb_decision > FF_MB_DECISION_SIMPLE){

        type= CANDIDATE_MB_TYPE_FORWARD | CANDIDATE_MB_TYPE_BACKWARD | CANDIDATE_MB_TYPE_BIDIR | CANDIDATE_MB_TYPE_DIRECT;

        if(fimin < INT_MAX)

            type |= CANDIDATE_MB_TYPE_FORWARD_I;

        if(bimin < INT_MAX)

            type |= CANDIDATE_MB_TYPE_BACKWARD_I;

        if(fimin < INT_MAX && bimin < INT_MAX){

            type |= CANDIDATE_MB_TYPE_BIDIR_I;

        }

         //FIXME something smarter

        if(dmin>256*256*16) type&= ~CANDIDATE_MB_TYPE_DIRECT; //do not try direct mode if it is invalid for this MB

        if(s->codec_id == AV_CODEC_ID_MPEG4 && type&CANDIDATE_MB_TYPE_DIRECT && s->flags&CODEC_FLAG_MV0 && *(uint32_t*)s->b_direct_mv_table[xy])

            type |= CANDIDATE_MB_TYPE_DIRECT0;

    }



    s->mb_type[mb_y*s->mb_stride + mb_x]= type;

}
