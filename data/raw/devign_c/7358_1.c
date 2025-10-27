static void encode_picture(MpegEncContext *s, int picture_number)

{

    int mb_x, mb_y, last_gob, pdif = 0;

    int i;

    int bits;

    MpegEncContext best_s, backup_s;

    UINT8 bit_buf[7][3000]; //FIXME check that this is ALLWAYS large enogh for a MB



    s->picture_number = picture_number;



    s->block_wrap[0]=

    s->block_wrap[1]=

    s->block_wrap[2]=

    s->block_wrap[3]= s->mb_width*2 + 2;

    s->block_wrap[4]=

    s->block_wrap[5]= s->mb_width + 2;

    

    /* Reset the average MB variance */

    s->avg_mb_var = 0;

    s->mc_mb_var = 0;



    /* we need to initialize some time vars before we can encode b-frames */

    if (s->h263_pred && !s->h263_msmpeg4)

        ff_set_mpeg4_time(s, s->picture_number); 



    /* Estimate motion for every MB */

    if(s->pict_type != I_TYPE){

//        int16_t (*tmp)[2]= s->p_mv_table;

//        s->p_mv_table= s->last_mv_table;

//        s->last_mv_table= s->mv_table;

    

        for(mb_y=0; mb_y < s->mb_height; mb_y++) {

            s->block_index[0]= s->block_wrap[0]*(mb_y*2 + 1) - 1;

            s->block_index[1]= s->block_wrap[0]*(mb_y*2 + 1);

            s->block_index[2]= s->block_wrap[0]*(mb_y*2 + 2) - 1;

            s->block_index[3]= s->block_wrap[0]*(mb_y*2 + 2);

            for(mb_x=0; mb_x < s->mb_width; mb_x++) {

                s->mb_x = mb_x;

                s->mb_y = mb_y;

                s->block_index[0]+=2;

                s->block_index[1]+=2;

                s->block_index[2]+=2;

                s->block_index[3]+=2;



                /* compute motion vector & mb_type and store in context */

                if(s->pict_type==B_TYPE)

                    ff_estimate_b_frame_motion(s, mb_x, mb_y);

                else

                    ff_estimate_p_frame_motion(s, mb_x, mb_y);

//                s->mb_type[mb_y*s->mb_width + mb_x]=MB_TYPE_INTER;

            }

        }

        emms_c();

    }else if(s->pict_type == I_TYPE){

        /* I-Frame */

        //FIXME do we need to zero them?

        memset(s->motion_val[0], 0, sizeof(INT16)*(s->mb_width*2 + 2)*(s->mb_height*2 + 2)*2);

        memset(s->p_mv_table   , 0, sizeof(INT16)*(s->mb_width+2)*(s->mb_height+2)*2);

        memset(s->mb_type      , MB_TYPE_INTRA, sizeof(UINT8)*s->mb_width*s->mb_height);

    }



    if(s->avg_mb_var < s->mc_mb_var && s->pict_type == P_TYPE){ //FIXME subtract MV bits

        s->pict_type= I_TYPE;

        memset(s->mb_type   , MB_TYPE_INTRA, sizeof(UINT8)*s->mb_width*s->mb_height);

        if(s->max_b_frames==0){

            s->input_pict_type= I_TYPE;

            s->input_picture_in_gop_number=0;

        }

//printf("Scene change detected, encoding as I Frame\n");

    }

    

    if(s->pict_type==P_TYPE || s->pict_type==S_TYPE) 

        s->f_code= ff_get_best_fcode(s, s->p_mv_table, MB_TYPE_INTER);

        ff_fix_long_p_mvs(s);

    if(s->pict_type==B_TYPE){

        s->f_code= ff_get_best_fcode(s, s->b_forw_mv_table, MB_TYPE_FORWARD);

        s->b_code= ff_get_best_fcode(s, s->b_back_mv_table, MB_TYPE_BACKWARD);



        ff_fix_long_b_mvs(s, s->b_forw_mv_table, s->f_code, MB_TYPE_FORWARD);

        ff_fix_long_b_mvs(s, s->b_back_mv_table, s->b_code, MB_TYPE_BACKWARD);

        ff_fix_long_b_mvs(s, s->b_bidir_forw_mv_table, s->f_code, MB_TYPE_BIDIR);

        ff_fix_long_b_mvs(s, s->b_bidir_back_mv_table, s->b_code, MB_TYPE_BIDIR);

    }

    

//printf("f_code %d ///\n", s->f_code);



//    printf("%d %d\n", s->avg_mb_var, s->mc_mb_var);



    if(s->flags&CODEC_FLAG_PASS2)

        s->qscale = ff_rate_estimate_qscale_pass2(s);

    else if (!s->fixed_qscale) 

        s->qscale = ff_rate_estimate_qscale(s);





    /* precompute matrix */

    if (s->out_format == FMT_MJPEG) {

        /* for mjpeg, we do include qscale in the matrix */

        s->intra_matrix[0] = default_intra_matrix[0];

        for(i=1;i<64;i++)

            s->intra_matrix[i] = (default_intra_matrix[i] * s->qscale) >> 3;

        convert_matrix(s->q_intra_matrix, s->q_intra_matrix16, s->intra_matrix, 8);

    } else {

        convert_matrix(s->q_intra_matrix, s->q_intra_matrix16, s->intra_matrix, s->qscale);

        convert_matrix(s->q_non_intra_matrix, s->q_non_intra_matrix16, s->non_intra_matrix, s->qscale);

    }



    s->last_bits= get_bit_count(&s->pb);

    switch(s->out_format) {

    case FMT_MJPEG:

        mjpeg_picture_header(s);

        break;

    case FMT_H263:

        if (s->h263_msmpeg4) 

            msmpeg4_encode_picture_header(s, picture_number);

        else if (s->h263_pred)

            mpeg4_encode_picture_header(s, picture_number);

        else if (s->h263_rv10) 

            rv10_encode_picture_header(s, picture_number);

        else

            h263_encode_picture_header(s, picture_number);

        break;

    case FMT_MPEG1:

        mpeg1_encode_picture_header(s, picture_number);

        break;

    }

    bits= get_bit_count(&s->pb);

    s->header_bits= bits - s->last_bits;

    s->last_bits= bits;

    s->mv_bits=0;

    s->misc_bits=0;

    s->i_tex_bits=0;

    s->p_tex_bits=0;

    s->i_count=0;

    s->p_count=0;

    s->skip_count=0;



    /* init last dc values */

    /* note: quant matrix value (8) is implied here */

    s->last_dc[0] = 128;

    s->last_dc[1] = 128;

    s->last_dc[2] = 128;

    s->mb_incr = 1;

    s->last_mv[0][0][0] = 0;

    s->last_mv[0][0][1] = 0;



    /* Get the GOB height based on picture height */

    if (s->out_format == FMT_H263 && !s->h263_pred && !s->h263_msmpeg4) {

        if (s->height <= 400)

            s->gob_index = 1;

        else if (s->height <= 800)

            s->gob_index = 2;

        else

            s->gob_index = 4;

    }

        

    s->avg_mb_var = s->avg_mb_var / s->mb_num;        

    

    for(mb_y=0; mb_y < s->mb_height; mb_y++) {

        /* Put GOB header based on RTP MTU */

        /* TODO: Put all this stuff in a separate generic function */

        if (s->rtp_mode) {

            if (!mb_y) {

                s->ptr_lastgob = s->pb.buf;

                s->ptr_last_mb_line = s->pb.buf;

            } else if (s->out_format == FMT_H263 && !s->h263_pred && !s->h263_msmpeg4 && !(mb_y % s->gob_index)) {

                last_gob = h263_encode_gob_header(s, mb_y);

                if (last_gob) {

                    s->first_gob_line = 1;

                }

            }

        }

        

        s->block_index[0]= s->block_wrap[0]*(mb_y*2 + 1) - 1;

        s->block_index[1]= s->block_wrap[0]*(mb_y*2 + 1);

        s->block_index[2]= s->block_wrap[0]*(mb_y*2 + 2) - 1;

        s->block_index[3]= s->block_wrap[0]*(mb_y*2 + 2);

        s->block_index[4]= s->block_wrap[4]*(mb_y + 1)                    + s->block_wrap[0]*(s->mb_height*2 + 2);

        s->block_index[5]= s->block_wrap[4]*(mb_y + 1 + s->mb_height + 2) + s->block_wrap[0]*(s->mb_height*2 + 2);

        for(mb_x=0; mb_x < s->mb_width; mb_x++) {

            const int mb_type= s->mb_type[mb_y * s->mb_width + mb_x];

            const int xy= (mb_y+1) * (s->mb_width+2) + mb_x + 1;

            PutBitContext pb;

            int d;

            int dmin=10000000;

            int best=0;



            s->mb_x = mb_x;

            s->mb_y = mb_y;

            s->block_index[0]+=2;

            s->block_index[1]+=2;

            s->block_index[2]+=2;

            s->block_index[3]+=2;

            s->block_index[4]++;

            s->block_index[5]++;

            if(mb_type & (mb_type-1)){ // more than 1 MB type possible

                int next_block=0;

                pb= s->pb;



                copy_context_before_encode(&backup_s, s, -1);



                if(mb_type&MB_TYPE_INTER){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->p_mv_table[xy][0];

                    s->mv[0][0][1] = s->p_mv_table[xy][1];

                    init_put_bits(&s->pb, bit_buf[1], 3000, NULL, NULL);

                    s->block= s->blocks[next_block];

                    s->last_bits= 0; //done in copy_context_before_encode but we skip that here



                    encode_mb(s, s->mv[0][0][0], s->mv[0][0][1]);

                    d= get_bit_count(&s->pb);

                    if(d<dmin){

                        flush_put_bits(&s->pb);

                        dmin=d;

                        copy_context_after_encode(&best_s, s, MB_TYPE_INTER);

                        best=1;

                        next_block^=1;

                    }

                }

                if(mb_type&MB_TYPE_INTER4V){                 

                    copy_context_before_encode(s, &backup_s, MB_TYPE_INTER4V);

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_8X8;

                    s->mb_intra= 0;

                    for(i=0; i<4; i++){

                        s->mv[0][i][0] = s->motion_val[s->block_index[i]][0];

                        s->mv[0][i][1] = s->motion_val[s->block_index[i]][1];

                    }

                    init_put_bits(&s->pb, bit_buf[2], 3000, NULL, NULL);

                    s->block= s->blocks[next_block];



                    encode_mb(s, 0, 0);

                    d= get_bit_count(&s->pb);

                    if(d<dmin){

                        flush_put_bits(&s->pb);

                        dmin=d;

                        copy_context_after_encode(&best_s, s, MB_TYPE_INTER4V);

                        best=2;

                        next_block^=1;

                    }

                }

                if(mb_type&MB_TYPE_FORWARD){

                    copy_context_before_encode(s, &backup_s, MB_TYPE_FORWARD);

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->b_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_forw_mv_table[xy][1];

                    init_put_bits(&s->pb, bit_buf[3], 3000, NULL, NULL);

                    s->block= s->blocks[next_block];



                    encode_mb(s, s->mv[0][0][0], s->mv[0][0][1]);

                    d= get_bit_count(&s->pb);

                    if(d<dmin){

                        flush_put_bits(&s->pb);

                        dmin=d;

                        copy_context_after_encode(&best_s, s, MB_TYPE_FORWARD);

                        best=3;

                        next_block^=1;

                    }

                }

                if(mb_type&MB_TYPE_BACKWARD){

                    copy_context_before_encode(s, &backup_s, MB_TYPE_BACKWARD);

                    s->mv_dir = MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[1][0][0] = s->b_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_back_mv_table[xy][1];

                    init_put_bits(&s->pb, bit_buf[4], 3000, NULL, NULL);

                    s->block= s->blocks[next_block];



                    encode_mb(s, s->mv[1][0][0], s->mv[1][0][1]);

                    d= get_bit_count(&s->pb);

                    if(d<dmin){

                        flush_put_bits(&s->pb);

                        dmin=d;

                        copy_context_after_encode(&best_s, s, MB_TYPE_BACKWARD);

                        best=4;

                        next_block^=1;

                    }

                }

                if(mb_type&MB_TYPE_BIDIR){

                    copy_context_before_encode(s, &backup_s, MB_TYPE_BIDIR);

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->b_bidir_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_bidir_forw_mv_table[xy][1];

                    s->mv[1][0][0] = s->b_bidir_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_bidir_back_mv_table[xy][1];

                    init_put_bits(&s->pb, bit_buf[5], 3000, NULL, NULL);

                    s->block= s->blocks[next_block];



                    encode_mb(s, 0, 0);

                    d= get_bit_count(&s->pb);

                    if(d<dmin){

                        flush_put_bits(&s->pb);

                        dmin=d;

                        copy_context_after_encode(&best_s, s, MB_TYPE_BIDIR);

                        best=5;

                        next_block^=1;

                    }

                }

                if(mb_type&MB_TYPE_DIRECT){

                    copy_context_before_encode(s, &backup_s, MB_TYPE_DIRECT);

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;

                    s->mv_type = MV_TYPE_16X16; //FIXME

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->b_direct_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_direct_forw_mv_table[xy][1];

                    s->mv[1][0][0] = s->b_direct_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_direct_back_mv_table[xy][1];

                    init_put_bits(&s->pb, bit_buf[6], 3000, NULL, NULL);

                    s->block= s->blocks[next_block];



                    encode_mb(s, s->b_direct_mv_table[xy][0], s->b_direct_mv_table[xy][1]);

                    d= get_bit_count(&s->pb);

                    if(d<dmin){

                        flush_put_bits(&s->pb);

                        dmin=d;

                        copy_context_after_encode(&best_s, s, MB_TYPE_DIRECT);

                        best=6;

                        next_block^=1;

                    }

                }

                if(mb_type&MB_TYPE_INTRA){

                    copy_context_before_encode(s, &backup_s, MB_TYPE_INTRA);

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 1;

                    s->mv[0][0][0] = 0;

                    s->mv[0][0][1] = 0;

                    init_put_bits(&s->pb, bit_buf[0], 3000, NULL, NULL);

                    s->block= s->blocks[next_block];

                   

                    encode_mb(s, 0, 0);

                    d= get_bit_count(&s->pb);

                    if(d<dmin){

                        flush_put_bits(&s->pb);

                        dmin=d;

                        copy_context_after_encode(&best_s, s, MB_TYPE_INTRA);

                        best=0;

                        next_block^=1;

                    }

                    /* force cleaning of ac/dc pred stuff if needed ... */

                    if(s->h263_pred || s->h263_aic)

                        s->mbintra_table[mb_x + mb_y*s->mb_width]=1;

                }

                copy_context_after_encode(s, &best_s, -1);

                copy_bits(&pb, bit_buf[best], dmin);

                s->pb= pb;

                s->last_bits= get_bit_count(&s->pb);

            } else {

                int motion_x, motion_y;

                s->mv_type=MV_TYPE_16X16;

                // only one MB-Type possible

                switch(mb_type){

                case MB_TYPE_INTRA:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mb_intra= 1;

                    motion_x= s->mv[0][0][0] = 0;

                    motion_y= s->mv[0][0][1] = 0;

                    break;

                case MB_TYPE_INTER:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[0][0][0] = s->p_mv_table[xy][0];

                    motion_y= s->mv[0][0][1] = s->p_mv_table[xy][1];

                    break;

                case MB_TYPE_DIRECT:

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;

                    s->mb_intra= 0;

                    motion_x=s->b_direct_mv_table[xy][0];

                    motion_y=s->b_direct_mv_table[xy][1];

                    s->mv[0][0][0] = s->b_direct_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_direct_forw_mv_table[xy][1];

                    s->mv[1][0][0] = s->b_direct_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_direct_back_mv_table[xy][1];

                    break;

                case MB_TYPE_BIDIR:

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;

                    s->mb_intra= 0;

                    motion_x=0;

                    motion_y=0;

                    s->mv[0][0][0] = s->b_bidir_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_bidir_forw_mv_table[xy][1];

                    s->mv[1][0][0] = s->b_bidir_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_bidir_back_mv_table[xy][1];

                    break;

                case MB_TYPE_BACKWARD:

                    s->mv_dir = MV_DIR_BACKWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[1][0][0] = s->b_back_mv_table[xy][0];

                    motion_y= s->mv[1][0][1] = s->b_back_mv_table[xy][1];

                    break;

                case MB_TYPE_FORWARD:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[0][0][0] = s->b_forw_mv_table[xy][0];

                    motion_y= s->mv[0][0][1] = s->b_forw_mv_table[xy][1];

//                    printf(" %d %d ", motion_x, motion_y);

                    break;

                default:

                    motion_x=motion_y=0; //gcc warning fix

                    printf("illegal MB type\n");

                }

                encode_mb(s, motion_x, motion_y);

            }

            /* clean the MV table in IPS frames for direct mode in B frames */

            if(s->mb_intra /* && I,P,S_TYPE */){

                s->p_mv_table[xy][0]=0;

                s->p_mv_table[xy][1]=0;

            }



            MPV_decode_mb(s, s->block);

        }





        /* Obtain average GOB size for RTP */

        if (s->rtp_mode) {

            if (!mb_y)

                s->mb_line_avgsize = pbBufPtr(&s->pb) - s->ptr_last_mb_line;

            else if (!(mb_y % s->gob_index)) {    

                s->mb_line_avgsize = (s->mb_line_avgsize + pbBufPtr(&s->pb) - s->ptr_last_mb_line) >> 1;

                s->ptr_last_mb_line = pbBufPtr(&s->pb);

            }

            //fprintf(stderr, "\nMB line: %d\tSize: %u\tAvg. Size: %u", s->mb_y, 

            //                    (s->pb.buf_ptr - s->ptr_last_mb_line), s->mb_line_avgsize);

            s->first_gob_line = 0;

        }

    }

    emms_c();



    if (s->h263_msmpeg4 && s->msmpeg4_version<4 && s->pict_type == I_TYPE)

        msmpeg4_encode_ext_header(s);



    //if (s->gob_number)

    //    fprintf(stderr,"\nNumber of GOB: %d", s->gob_number);

    

    /* Send the last GOB if RTP */    

    if (s->rtp_mode) {

        flush_put_bits(&s->pb);

        pdif = pbBufPtr(&s->pb) - s->ptr_lastgob;

        /* Call the RTP callback to send the last GOB */

        if (s->rtp_callback)

            s->rtp_callback(s->ptr_lastgob, pdif, s->gob_number);

        s->ptr_lastgob = pbBufPtr(&s->pb);

        //fprintf(stderr,"\nGOB: %2d size: %d (last)", s->gob_number, pdif);

    }

}
