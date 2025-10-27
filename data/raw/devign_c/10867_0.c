static int encode_frame(AVCodecContext *avctx, unsigned char *buf, int buf_size, void *data){

    SnowContext *s = avctx->priv_data;

    CABACContext * const c= &s->c;

    AVFrame *pict = data;

    const int width= s->avctx->width;

    const int height= s->avctx->height;

    int used_count= 0;

    int log2_threshold, level, orientation, plane_index, i;



    ff_init_cabac_encoder(c, buf, buf_size);

    ff_init_cabac_states(c, ff_h264_lps_range, ff_h264_mps_state, ff_h264_lps_state, 64);

    

    s->input_picture = *pict;



    memset(s->header_state, 0, sizeof(s->header_state));



    s->keyframe=avctx->gop_size==0 || avctx->frame_number % avctx->gop_size == 0;

    pict->pict_type= s->keyframe ? FF_I_TYPE : FF_P_TYPE;

    

    if(pict->quality){

        s->qlog= rint(QROOT*log(pict->quality / (float)FF_QP2LAMBDA)/log(2));

        //<64 >60

        s->qlog += 61;

    }else{

        s->qlog= LOSSLESS_QLOG;

    }



    for(i=0; i<s->mb_band.stride * s->mb_band.height; i++){

        s->mb_band.buf[i]= s->keyframe;

    }

    

    frame_start(s);



    if(pict->pict_type == P_TYPE){

        int block_width = (width +15)>>4;

        int block_height= (height+15)>>4;

        int stride= s->current_picture.linesize[0];

        uint8_t *src_plane= s->input_picture.data[0];

        int src_stride= s->input_picture.linesize[0];

        int x,y;

        

        assert(s->current_picture.data[0]);

        assert(s->last_picture.data[0]);

     

        s->m.avctx= s->avctx;

        s->m.current_picture.data[0]= s->current_picture.data[0];

        s->m.   last_picture.data[0]= s->   last_picture.data[0];

        s->m.    new_picture.data[0]= s->  input_picture.data[0];

        s->m.current_picture_ptr= &s->m.current_picture;

        s->m.   last_picture_ptr= &s->m.   last_picture;

        s->m.linesize=

        s->m.   last_picture.linesize[0]=

        s->m.    new_picture.linesize[0]=

        s->m.current_picture.linesize[0]= stride;

        s->m.width = width;

        s->m.height= height;

        s->m.mb_width = block_width;

        s->m.mb_height= block_height;

        s->m.mb_stride=   s->m.mb_width+1;

        s->m.b8_stride= 2*s->m.mb_width+1;

        s->m.f_code=1;

        s->m.pict_type= pict->pict_type;

        s->m.me_method= s->avctx->me_method;

        s->m.me.scene_change_score=0;

        s->m.flags= s->avctx->flags;

        s->m.quarter_sample= (s->avctx->flags & CODEC_FLAG_QPEL)!=0;

        s->m.out_format= FMT_H263;

        s->m.unrestricted_mv= 1;



        s->m.lambda= pict->quality * 3/2; //FIXME bug somewhere else

        s->m.qscale= (s->m.lambda*139 + FF_LAMBDA_SCALE*64) >> (FF_LAMBDA_SHIFT + 7);

        s->m.lambda2= (s->m.lambda*s->m.lambda + FF_LAMBDA_SCALE/2) >> FF_LAMBDA_SHIFT;



        if(!s->motion_val8){

            s->motion_val8 = av_mallocz(s->m.b8_stride*block_height*2*2*sizeof(int16_t));

            s->motion_val16= av_mallocz(s->m.mb_stride*block_height*2*sizeof(int16_t));

        }

        

        s->m.mb_type= s->mb_type;

        

        //dummies, to avoid segfaults

        s->m.current_picture.mb_mean  = s->mb_mean;

        s->m.current_picture.mb_var   = (int16_t*)s->dummy;

        s->m.current_picture.mc_mb_var= (int16_t*)s->dummy;

        s->m.current_picture.mb_type  = s->dummy;

        

        s->m.current_picture.motion_val[0]= s->motion_val8;

        s->m.p_mv_table= s->motion_val16;

        s->m.dsp= s->dsp; //move

        ff_init_me(&s->m);

    

        

        s->m.me.pre_pass=1;

        s->m.me.dia_size= s->avctx->pre_dia_size;

        s->m.first_slice_line=1;

        for(y= block_height-1; y >= 0; y--) {

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



            for(x=block_width-1; x >=0 ;x--) {

                s->m.mb_x= x;

                ff_init_block_index(&s->m);

                ff_update_block_index(&s->m);

                ff_pre_estimate_p_frame_motion(&s->m, x, y);

            }

            s->m.first_slice_line=0;

        }        

        s->m.me.pre_pass=0;

        

        

        s->m.me.dia_size= s->avctx->dia_size;

        s->m.first_slice_line=1;

        for (y = 0; y < block_height; y++) {

            uint8_t src[stride*16];



            s->m.new_picture.data[0]= src - y*16*stride; //ugly

            s->m.mb_y= y;

            

            assert(width <= stride);

            assert(width <= 16*block_width);

    

            for(i=0; i<16 && i + 16*y<height; i++){

                memcpy(&src[i*stride], &src_plane[(i+16*y)*src_stride], width);

                for(x=width; x<16*block_width; x++)

                    src[i*stride+x]= src[i*stride+x-1];

            }

            for(; i<16 && i + 16*y<16*block_height; i++)

                memcpy(&src[i*stride], &src[(i-1)*stride], 16*block_width);

    

            for (x = 0; x < block_width; x++) {

                int mb_xy= x + y*(s->mb_band.stride);

                s->m.mb_x= x;

                ff_init_block_index(&s->m);

                ff_update_block_index(&s->m);

                

                ff_estimate_p_frame_motion(&s->m, x, y);

                

                s->mb_band   .buf[mb_xy]= (s->m.mb_type[x + y*s->m.mb_stride]&CANDIDATE_MB_TYPE_INTER)

                 ? 0 : 2;

                s->mv_band[0].buf[mb_xy]= s->motion_val16[x + y*s->m.mb_stride][0];

                s->mv_band[1].buf[mb_xy]= s->motion_val16[x + y*s->m.mb_stride][1];

                

                if(s->mb_band   .buf[x + y*(s->mb_band.stride)]==2 && 0){

                    int dc0=128, dc1=128, dc, dc2, dir;

                    int offset= (s->avctx->flags & CODEC_FLAG_QPEL) ? 64 : 32;

                    

                    dc       =s->mb_mean[x +  y   *s->m.mb_stride    ];

                    if(x) dc0=s->mb_mean[x +  y   *s->m.mb_stride - 1];

                    if(y) dc1=s->mb_mean[x + (y-1)*s->m.mb_stride    ];

                    dc2= (dc0+dc1)>>1;

#if 0

                    if     (ABS(dc0 - dc) < ABS(dc1 - dc) && ABS(dc0 - dc) < ABS(dc2 - dc))

                        dir= 1;

                    else if(ABS(dc0 - dc) >=ABS(dc1 - dc) && ABS(dc1 - dc) < ABS(dc2 - dc))

                        dir=-1;

                    else

                        dir=0;

#endif                    

                    if(ABS(dc0 - dc) < ABS(dc1 - dc) && x){

                        s->mv_band[0].buf[mb_xy]= s->mv_band[0].buf[x + y*(s->mb_band.stride)-1] - offset;

                        s->mv_band[1].buf[mb_xy]= s->mv_band[1].buf[x + y*(s->mb_band.stride)-1];

                        s->mb_mean[x +  y   *s->m.mb_stride    ]= dc0;

                    }else if(y){

                        s->mv_band[0].buf[mb_xy]= s->mv_band[0].buf[x + (y-1)*(s->mb_band.stride)];

                        s->mv_band[1].buf[mb_xy]= s->mv_band[1].buf[x + (y-1)*(s->mb_band.stride)] - offset;

                        s->mb_mean[x +  y   *s->m.mb_stride    ]= dc1;

                    }

                }

//                s->mb_band   .buf[x + y*(s->mb_band.stride)]=1; //FIXME intra only test

            }

            s->m.first_slice_line=0;

        }

        assert(s->m.pict_type == P_TYPE);

        if(s->m.me.scene_change_score > s->avctx->scenechange_threshold){

            s->m.pict_type= 

            pict->pict_type =I_TYPE;

            for(i=0; i<s->mb_band.stride * s->mb_band.height; i++){

                s->mb_band.buf[i]= 1;

                s->mv_band[0].buf[i]=

                s->mv_band[1].buf[i]= 0;

            }

    //printf("Scene change detected, encoding as I Frame %d %d\n", s->current_picture.mb_var_sum, s->current_picture.mc_mb_var_sum);

        }        

    }

        

    s->m.first_slice_line=1;

    

    s->qbias= pict->pict_type == P_TYPE ? 2 : 0;



    encode_header(s);

    

    decorrelate(s, &s->mb_band   , s->mb_band   .buf, s->mb_band   .stride, 0, 1);

    decorrelate(s, &s->mv_band[0], s->mv_band[0].buf, s->mv_band[0].stride, 0, 1);

    decorrelate(s, &s->mv_band[1], s->mv_band[1].buf, s->mv_band[1].stride, 0 ,1);

    encode_subband(s, &s->mb_band   , s->mb_band   .buf, NULL, s->mb_band   .stride, 0);

    encode_subband(s, &s->mv_band[0], s->mv_band[0].buf, NULL, s->mv_band[0].stride, 0);

    encode_subband(s, &s->mv_band[1], s->mv_band[1].buf, NULL, s->mv_band[1].stride, 0);

    

//FIXME avoid this

    correlate(s, &s->mb_band   , s->mb_band   .buf, s->mb_band   .stride, 1, 1);

    correlate(s, &s->mv_band[0], s->mv_band[0].buf, s->mv_band[0].stride, 1, 1);

    correlate(s, &s->mv_band[1], s->mv_band[1].buf, s->mv_band[1].stride, 1, 1);

    

    for(plane_index=0; plane_index<3; plane_index++){

        Plane *p= &s->plane[plane_index];

        int w= p->width;

        int h= p->height;

        int x, y;

        int bits= put_bits_count(&s->c.pb);



        //FIXME optimize

     if(pict->data[plane_index]) //FIXME gray hack

        for(y=0; y<h; y++){

            for(x=0; x<w; x++){

                s->spatial_dwt_buffer[y*w + x]= pict->data[plane_index][y*pict->linesize[plane_index] + x]<<8;

            }

        }

        predict_plane(s, s->spatial_dwt_buffer, plane_index, 0);

        if(s->qlog == LOSSLESS_QLOG){

            for(y=0; y<h; y++){

                for(x=0; x<w; x++){

                    s->spatial_dwt_buffer[y*w + x]= (s->spatial_dwt_buffer[y*w + x] + 127)>>8;

                }

            }

        }

 

        ff_spatial_dwt(s->spatial_dwt_buffer, w, h, w, s->spatial_decomposition_type, s->spatial_decomposition_count);



        for(level=0; level<s->spatial_decomposition_count; level++){

            for(orientation=level ? 1 : 0; orientation<4; orientation++){

                SubBand *b= &p->band[level][orientation];

                

                quantize(s, b, b->buf, b->stride, s->qbias);

                if(orientation==0)

                    decorrelate(s, b, b->buf, b->stride, pict->pict_type == P_TYPE, 0);

                encode_subband(s, b, b->buf, b->parent ? b->parent->buf : NULL, b->stride, orientation);

                assert(b->parent==NULL || b->parent->stride == b->stride*2);

                if(orientation==0)

                    correlate(s, b, b->buf, b->stride, 1, 0);

            }

        }

//        av_log(NULL, AV_LOG_DEBUG, "plane:%d bits:%d\n", plane_index, put_bits_count(&s->c.pb) - bits);



        for(level=0; level<s->spatial_decomposition_count; level++){

            for(orientation=level ? 1 : 0; orientation<4; orientation++){

                SubBand *b= &p->band[level][orientation];



                dequantize(s, b, b->buf, b->stride);

            }

        }



        ff_spatial_idwt(s->spatial_dwt_buffer, w, h, w, s->spatial_decomposition_type, s->spatial_decomposition_count);

        if(s->qlog == LOSSLESS_QLOG){

            for(y=0; y<h; y++){

                for(x=0; x<w; x++){

                    s->spatial_dwt_buffer[y*w + x]<<=8;

                }

            }

        }

        predict_plane(s, s->spatial_dwt_buffer, plane_index, 1);

        //FIXME optimize

        for(y=0; y<h; y++){

            for(x=0; x<w; x++){

                int v= (s->spatial_dwt_buffer[y*w + x]+128)>>8;

                if(v&(~255)) v= ~(v>>31);

                s->current_picture.data[plane_index][y*s->current_picture.linesize[plane_index] + x]= v;

            }

        }

        if(s->avctx->flags&CODEC_FLAG_PSNR){

            int64_t error= 0;

            

    if(pict->data[plane_index]) //FIXME gray hack

            for(y=0; y<h; y++){

                for(x=0; x<w; x++){

                    int d= s->current_picture.data[plane_index][y*s->current_picture.linesize[plane_index] + x] - pict->data[plane_index][y*pict->linesize[plane_index] + x];

                    error += d*d;

                }

            }

            s->avctx->error[plane_index] += error;

            s->avctx->error[3] += error;

        }

    }



    if(s->last_picture.data[0])

        avctx->release_buffer(avctx, &s->last_picture);



    emms_c();

    

    return put_cabac_terminate(c, 1);

}
