void mpeg1_encode_mb(MpegEncContext *s,

                     DCTELEM block[6][64],

                     int motion_x, int motion_y)

{

    int i, cbp;

    const int mb_x = s->mb_x;

    const int mb_y = s->mb_y;

    const int first_mb= mb_x == s->resync_mb_x && mb_y == s->resync_mb_y;



    /* compute cbp */

    cbp = 0;

    for(i=0;i<6;i++) {

        if (s->block_last_index[i] >= 0)

            cbp |= 1 << (5 - i);

    }

    

    if (cbp == 0 && !first_mb && (mb_x != s->mb_width - 1 || (mb_y != s->mb_height - 1 && s->codec_id == CODEC_ID_MPEG1VIDEO)) && 

        ((s->pict_type == P_TYPE && s->mv_type == MV_TYPE_16X16 && (motion_x | motion_y) == 0) ||

        (s->pict_type == B_TYPE && s->mv_dir == s->last_mv_dir && (((s->mv_dir & MV_DIR_FORWARD) ? ((s->mv[0][0][0] - s->last_mv[0][0][0])|(s->mv[0][0][1] - s->last_mv[0][0][1])) : 0) |

        ((s->mv_dir & MV_DIR_BACKWARD) ? ((s->mv[1][0][0] - s->last_mv[1][0][0])|(s->mv[1][0][1] - s->last_mv[1][0][1])) : 0)) == 0))) {

        s->mb_skip_run++;

        s->qscale -= s->dquant;

        s->skip_count++;

        s->misc_bits++;

        s->last_bits++;

        if(s->pict_type == P_TYPE){

            s->last_mv[0][1][0]= s->last_mv[0][0][0]= 

            s->last_mv[0][1][1]= s->last_mv[0][0][1]= 0;

        }

    } else {

        if(first_mb){

            assert(s->mb_skip_run == 0);

            encode_mb_skip_run(s, s->mb_x);

        }else{

            encode_mb_skip_run(s, s->mb_skip_run);

        }

        

        if (s->pict_type == I_TYPE) {

            if(s->dquant && cbp){

                put_mb_modes(s, 2, 1, 0, 0); /* macroblock_type : macroblock_quant = 1 */

                put_bits(&s->pb, 5, s->qscale);

            }else{

                put_mb_modes(s, 1, 1, 0, 0); /* macroblock_type : macroblock_quant = 0 */

                s->qscale -= s->dquant;

            }

            s->misc_bits+= get_bits_diff(s);

            s->i_count++;

        } else if (s->mb_intra) {

            if(s->dquant && cbp){

                put_mb_modes(s, 6, 0x01, 0, 0);

                put_bits(&s->pb, 5, s->qscale);

            }else{

                put_mb_modes(s, 5, 0x03, 0, 0);

                s->qscale -= s->dquant;

            }

            s->misc_bits+= get_bits_diff(s);

            s->i_count++;

            memset(s->last_mv, 0, sizeof(s->last_mv));

        } else if (s->pict_type == P_TYPE) { 

            if(s->mv_type == MV_TYPE_16X16){

                if (cbp != 0) {

                    if ((motion_x|motion_y) == 0) {

                        if(s->dquant){

                            put_mb_modes(s, 5, 1, 0, 0); /* macroblock_pattern & quant */

                            put_bits(&s->pb, 5, s->qscale);

                        }else{

                            put_mb_modes(s, 2, 1, 0, 0); /* macroblock_pattern only */

                        }

                        s->misc_bits+= get_bits_diff(s);

                    } else {

                        if(s->dquant){

                            put_mb_modes(s, 5, 2, 1, 0); /* motion + cbp */

                            put_bits(&s->pb, 5, s->qscale);

                        }else{

                            put_mb_modes(s, 1, 1, 1, 0); /* motion + cbp */

                        }

                        s->misc_bits+= get_bits_diff(s);

                        mpeg1_encode_motion(s, motion_x - s->last_mv[0][0][0], s->f_code);    // RAL: f_code parameter added

                        mpeg1_encode_motion(s, motion_y - s->last_mv[0][0][1], s->f_code);    // RAL: f_code parameter added

                        s->mv_bits+= get_bits_diff(s);

                    }

                } else {

                    put_bits(&s->pb, 3, 1); /* motion only */

                    if (!s->frame_pred_frame_dct)

                        put_bits(&s->pb, 2, 2); /* motion_type: frame */

                    s->misc_bits+= get_bits_diff(s);

                    mpeg1_encode_motion(s, motion_x - s->last_mv[0][0][0], s->f_code);    // RAL: f_code parameter added

                    mpeg1_encode_motion(s, motion_y - s->last_mv[0][0][1], s->f_code);    // RAL: f_code parameter added

                    s->qscale -= s->dquant;

                    s->mv_bits+= get_bits_diff(s);

                }

                s->last_mv[0][1][0]= s->last_mv[0][0][0]= motion_x;

                s->last_mv[0][1][1]= s->last_mv[0][0][1]= motion_y;

            }else{

                assert(!s->frame_pred_frame_dct && s->mv_type == MV_TYPE_FIELD);



                if (cbp) {

                    if(s->dquant){

                        put_mb_modes(s, 5, 2, 1, 1); /* motion + cbp */

                        put_bits(&s->pb, 5, s->qscale);

                    }else{

                        put_mb_modes(s, 1, 1, 1, 1); /* motion + cbp */

                    }

                } else {

                    put_bits(&s->pb, 3, 1); /* motion only */

                    put_bits(&s->pb, 2, 1); /* motion_type: field */

                    s->qscale -= s->dquant;

                }

                s->misc_bits+= get_bits_diff(s);

                for(i=0; i<2; i++){

                    put_bits(&s->pb, 1, s->field_select[0][i]);

                    mpeg1_encode_motion(s, s->mv[0][i][0] -  s->last_mv[0][i][0]    , s->f_code);

                    mpeg1_encode_motion(s, s->mv[0][i][1] - (s->last_mv[0][i][1]>>1), s->f_code);

                    s->last_mv[0][i][0]=   s->mv[0][i][0];

                    s->last_mv[0][i][1]= 2*s->mv[0][i][1];

                }

                s->mv_bits+= get_bits_diff(s);

            }

            if(cbp)

                put_bits(&s->pb, mbPatTable[cbp - 1][1], mbPatTable[cbp - 1][0]);

            s->f_count++;

        } else{  

            static const int mb_type_len[4]={0,3,4,2}; //bak,for,bi



            if(s->mv_type == MV_TYPE_16X16){

                if (cbp){    // With coded bloc pattern

                    if (s->dquant) {

                        if(s->mv_dir == MV_DIR_FORWARD)

                            put_mb_modes(s, 6, 3, 1, 0);

                        else

                            put_mb_modes(s, mb_type_len[s->mv_dir]+3, 2, 1, 0);

                        put_bits(&s->pb, 5, s->qscale);

                    } else {

                        put_mb_modes(s, mb_type_len[s->mv_dir], 3, 1, 0);

                    }

                }else{    // No coded bloc pattern

                    put_bits(&s->pb, mb_type_len[s->mv_dir], 2);

                    if (!s->frame_pred_frame_dct)

                        put_bits(&s->pb, 2, 2); /* motion_type: frame */

                    s->qscale -= s->dquant;

                }

                s->misc_bits += get_bits_diff(s);

                if (s->mv_dir&MV_DIR_FORWARD){

                    mpeg1_encode_motion(s, s->mv[0][0][0] - s->last_mv[0][0][0], s->f_code); 

                    mpeg1_encode_motion(s, s->mv[0][0][1] - s->last_mv[0][0][1], s->f_code); 

                    s->last_mv[0][0][0]=s->last_mv[0][1][0]= s->mv[0][0][0];

                    s->last_mv[0][0][1]=s->last_mv[0][1][1]= s->mv[0][0][1];

                    s->f_count++;

                }

                if (s->mv_dir&MV_DIR_BACKWARD){

                    mpeg1_encode_motion(s, s->mv[1][0][0] - s->last_mv[1][0][0], s->b_code); 

                    mpeg1_encode_motion(s, s->mv[1][0][1] - s->last_mv[1][0][1], s->b_code); 

                    s->last_mv[1][0][0]=s->last_mv[1][1][0]= s->mv[1][0][0];

                    s->last_mv[1][0][1]=s->last_mv[1][1][1]= s->mv[1][0][1];

                    s->b_count++;

                }

            }else{

                assert(s->mv_type == MV_TYPE_FIELD);

                assert(!s->frame_pred_frame_dct);

                if (cbp){    // With coded bloc pattern

                    if (s->dquant) {

                        if(s->mv_dir == MV_DIR_FORWARD)

                            put_mb_modes(s, 6, 3, 1, 1);

                        else

                            put_mb_modes(s, mb_type_len[s->mv_dir]+3, 2, 1, 1);

                        put_bits(&s->pb, 5, s->qscale);

                    } else {

                        put_mb_modes(s, mb_type_len[s->mv_dir], 3, 1, 1);

                    }

                }else{    // No coded bloc pattern

                    put_bits(&s->pb, mb_type_len[s->mv_dir], 2);

                    put_bits(&s->pb, 2, 1); /* motion_type: field */

                    s->qscale -= s->dquant;

                }

                s->misc_bits += get_bits_diff(s);

                if (s->mv_dir&MV_DIR_FORWARD){

                    for(i=0; i<2; i++){

                        put_bits(&s->pb, 1, s->field_select[0][i]);

                        mpeg1_encode_motion(s, s->mv[0][i][0] -  s->last_mv[0][i][0]    , s->f_code);

                        mpeg1_encode_motion(s, s->mv[0][i][1] - (s->last_mv[0][i][1]>>1), s->f_code);

                        s->last_mv[0][i][0]=   s->mv[0][i][0];

                        s->last_mv[0][i][1]= 2*s->mv[0][i][1];

                    }

                    s->f_count++;

                }

                if (s->mv_dir&MV_DIR_BACKWARD){

                    for(i=0; i<2; i++){

                        put_bits(&s->pb, 1, s->field_select[1][i]);

                        mpeg1_encode_motion(s, s->mv[1][i][0] -  s->last_mv[1][i][0]    , s->b_code);

                        mpeg1_encode_motion(s, s->mv[1][i][1] - (s->last_mv[1][i][1]>>1), s->b_code);

                        s->last_mv[1][i][0]=   s->mv[1][i][0];

                        s->last_mv[1][i][1]= 2*s->mv[1][i][1];

                    }

                    s->b_count++;

                }

            }

            s->mv_bits += get_bits_diff(s);

            if(cbp)

                put_bits(&s->pb, mbPatTable[cbp - 1][1], mbPatTable[cbp - 1][0]);

        }

        for(i=0;i<6;i++) {

            if (cbp & (1 << (5 - i))) {

                mpeg1_encode_block(s, block[i], i);

            }

        }

        s->mb_skip_run = 0;

        if(s->mb_intra)

            s->i_tex_bits+= get_bits_diff(s);

        else

            s->p_tex_bits+= get_bits_diff(s);

    }

}
