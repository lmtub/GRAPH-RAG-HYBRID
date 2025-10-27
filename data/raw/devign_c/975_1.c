static int svq1_encode_plane(SVQ1Context *s, int plane, unsigned char *src_plane, unsigned char *ref_plane, unsigned char *decoded_plane,

    int width, int height, int src_stride, int stride)

{

    int x, y;

    int i;

    int block_width, block_height;

    int level;

    int threshold[6];

    const int lambda= (s->picture.quality*s->picture.quality) >> (2*FF_LAMBDA_SHIFT);



    /* figure out the acceptable level thresholds in advance */

    threshold[5] = QUALITY_THRESHOLD;

    for (level = 4; level >= 0; level--)

        threshold[level] = threshold[level + 1] * THRESHOLD_MULTIPLIER;



    block_width = (width + 15) / 16;

    block_height = (height + 15) / 16;



    if(s->picture.pict_type == FF_P_TYPE){

        s->m.avctx= s->avctx;

        s->m.current_picture_ptr= &s->m.current_picture;

        s->m.last_picture_ptr   = &s->m.last_picture;

        s->m.last_picture.data[0]= ref_plane;

        s->m.linesize=

        s->m.last_picture.linesize[0]=

        s->m.new_picture.linesize[0]=

        s->m.current_picture.linesize[0]= stride;

        s->m.width= width;

        s->m.height= height;

        s->m.mb_width= block_width;

        s->m.mb_height= block_height;

        s->m.mb_stride= s->m.mb_width+1;

        s->m.b8_stride= 2*s->m.mb_width+1;

        s->m.f_code=1;

        s->m.pict_type= s->picture.pict_type;

        s->m.me_method= s->avctx->me_method;

        s->m.me.scene_change_score=0;

        s->m.flags= s->avctx->flags;

//        s->m.out_format = FMT_H263;

//        s->m.unrestricted_mv= 1;



        s->m.lambda= s->picture.quality;

        s->m.qscale= (s->m.lambda*139 + FF_LAMBDA_SCALE*64) >> (FF_LAMBDA_SHIFT + 7);

        s->m.lambda2= (s->m.lambda*s->m.lambda + FF_LAMBDA_SCALE/2) >> FF_LAMBDA_SHIFT;



        if(!s->motion_val8[plane]){

            s->motion_val8 [plane]= av_mallocz((s->m.b8_stride*block_height*2 + 2)*2*sizeof(int16_t));

            s->motion_val16[plane]= av_mallocz((s->m.mb_stride*(block_height + 2) + 1)*2*sizeof(int16_t));

        }



        s->m.mb_type= s->mb_type;



        //dummies, to avoid segfaults

        s->m.current_picture.mb_mean=   (uint8_t *)s->dummy;

        s->m.current_picture.mb_var=    (uint16_t*)s->dummy;

        s->m.current_picture.mc_mb_var= (uint16_t*)s->dummy;

        s->m.current_picture.mb_type= s->dummy;



        s->m.current_picture.motion_val[0]= s->motion_val8[plane] + 2;

        s->m.p_mv_table= s->motion_val16[plane] + s->m.mb_stride + 1;

        s->m.dsp= s->dsp; //move

        ff_init_me(&s->m);



        s->m.me.dia_size= s->avctx->dia_size;

        s->m.first_slice_line=1;

        for (y = 0; y < block_height; y++) {

            uint8_t src[stride*16];



            s->m.new_picture.data[0]= src - y*16*stride; //ugly

            s->m.mb_y= y;



            for(i=0; i<16 && i + 16*y<height; i++){

                memcpy(&src[i*stride], &src_plane[(i+16*y)*src_stride], width);

                for(x=width; x<16*block_width; x++)

                    src[i*stride+x]= src[i*stride+x-1];

            }

            for(; i<16 && i + 16*y<16*block_height; i++)

                memcpy(&src[i*stride], &src[(i-1)*stride], 16*block_width);



            for (x = 0; x < block_width; x++) {

                s->m.mb_x= x;

                ff_init_block_index(&s->m);

                ff_update_block_index(&s->m);



                ff_estimate_p_frame_motion(&s->m, x, y);

            }

            s->m.first_slice_line=0;

        }



        ff_fix_long_p_mvs(&s->m);

        ff_fix_long_mvs(&s->m, NULL, 0, s->m.p_mv_table, s->m.f_code, CANDIDATE_MB_TYPE_INTER, 0);

    }



    s->m.first_slice_line=1;

    for (y = 0; y < block_height; y++) {

        uint8_t src[stride*16];



        for(i=0; i<16 && i + 16*y<height; i++){

            memcpy(&src[i*stride], &src_plane[(i+16*y)*src_stride], width);

            for(x=width; x<16*block_width; x++)

                src[i*stride+x]= src[i*stride+x-1];

        }

        for(; i<16 && i + 16*y<16*block_height; i++)

            memcpy(&src[i*stride], &src[(i-1)*stride], 16*block_width);



        s->m.mb_y= y;

        for (x = 0; x < block_width; x++) {

            uint8_t reorder_buffer[3][6][7*32];

            int count[3][6];

            int offset = y * 16 * stride + x * 16;

            uint8_t *decoded= decoded_plane + offset;

            uint8_t *ref= ref_plane + offset;

            int score[4]={0,0,0,0}, best;

            uint8_t *temp = s->scratchbuf;



            if(s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb)>>3) < 3000){ //FIXME check size

                av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");

                return -1;

            }



            s->m.mb_x= x;

            ff_init_block_index(&s->m);

            ff_update_block_index(&s->m);



            if(s->picture.pict_type == FF_I_TYPE || (s->m.mb_type[x + y*s->m.mb_stride]&CANDIDATE_MB_TYPE_INTRA)){

                for(i=0; i<6; i++){

                    init_put_bits(&s->reorder_pb[i], reorder_buffer[0][i], 7*32);

                }

                if(s->picture.pict_type == FF_P_TYPE){

                    const uint8_t *vlc= ff_svq1_block_type_vlc[SVQ1_BLOCK_INTRA];

                    put_bits(&s->reorder_pb[5], vlc[1], vlc[0]);

                    score[0]= vlc[1]*lambda;

                }

                score[0]+= encode_block(s, src+16*x, NULL, temp, stride, 5, 64, lambda, 1);

                for(i=0; i<6; i++){

                    count[0][i]= put_bits_count(&s->reorder_pb[i]);

                    flush_put_bits(&s->reorder_pb[i]);

                }

            }else

                score[0]= INT_MAX;



            best=0;



            if(s->picture.pict_type == FF_P_TYPE){

                const uint8_t *vlc= ff_svq1_block_type_vlc[SVQ1_BLOCK_INTER];

                int mx, my, pred_x, pred_y, dxy;

                int16_t *motion_ptr;



                motion_ptr= h263_pred_motion(&s->m, 0, 0, &pred_x, &pred_y);

                if(s->m.mb_type[x + y*s->m.mb_stride]&CANDIDATE_MB_TYPE_INTER){

                    for(i=0; i<6; i++)

                        init_put_bits(&s->reorder_pb[i], reorder_buffer[1][i], 7*32);



                    put_bits(&s->reorder_pb[5], vlc[1], vlc[0]);



                    s->m.pb= s->reorder_pb[5];

                    mx= motion_ptr[0];

                    my= motion_ptr[1];

                    assert(mx>=-32 && mx<=31);

                    assert(my>=-32 && my<=31);

                    assert(pred_x>=-32 && pred_x<=31);

                    assert(pred_y>=-32 && pred_y<=31);

                    ff_h263_encode_motion(&s->m, mx - pred_x, 1);

                    ff_h263_encode_motion(&s->m, my - pred_y, 1);

                    s->reorder_pb[5]= s->m.pb;

                    score[1] += lambda*put_bits_count(&s->reorder_pb[5]);



                    dxy= (mx&1) + 2*(my&1);



                    s->dsp.put_pixels_tab[0][dxy](temp+16, ref + (mx>>1) + stride*(my>>1), stride, 16);



                    score[1]+= encode_block(s, src+16*x, temp+16, decoded, stride, 5, 64, lambda, 0);

                    best= score[1] <= score[0];



                    vlc= ff_svq1_block_type_vlc[SVQ1_BLOCK_SKIP];

                    score[2]= s->dsp.sse[0](NULL, src+16*x, ref, stride, 16);

                    score[2]+= vlc[1]*lambda;

                    if(score[2] < score[best] && mx==0 && my==0){

                        best=2;

                        s->dsp.put_pixels_tab[0][0](decoded, ref, stride, 16);

                        for(i=0; i<6; i++){

                            count[2][i]=0;

                        }

                        put_bits(&s->pb, vlc[1], vlc[0]);

                    }

                }



                if(best==1){

                    for(i=0; i<6; i++){

                        count[1][i]= put_bits_count(&s->reorder_pb[i]);

                        flush_put_bits(&s->reorder_pb[i]);

                    }

                }else{

                    motion_ptr[0                 ] = motion_ptr[1                 ]=

                    motion_ptr[2                 ] = motion_ptr[3                 ]=

                    motion_ptr[0+2*s->m.b8_stride] = motion_ptr[1+2*s->m.b8_stride]=

                    motion_ptr[2+2*s->m.b8_stride] = motion_ptr[3+2*s->m.b8_stride]=0;

                }

            }



            s->rd_total += score[best];



            for(i=5; i>=0; i--){

                ff_copy_bits(&s->pb, reorder_buffer[best][i], count[best][i]);

            }

            if(best==0){

                s->dsp.put_pixels_tab[0][0](decoded, temp, stride, 16);

            }

        }

        s->m.first_slice_line=0;

    }

    return 0;

}
