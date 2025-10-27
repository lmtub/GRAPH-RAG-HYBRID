int ff_h263_decode_frame(AVCodecContext *avctx, 

                             void *data, int *data_size,

                             UINT8 *buf, int buf_size)

{

    MpegEncContext *s = avctx->priv_data;

    int ret,i;

    AVFrame *pict = data; 

    float new_aspect;

    

#ifdef PRINT_FRAME_TIME

uint64_t time= rdtsc();

#endif

#ifdef DEBUG

    printf("*****frame %d size=%d\n", avctx->frame_number, buf_size);

    printf("bytes=%x %x %x %x\n", buf[0], buf[1], buf[2], buf[3]);

#endif

    s->flags= avctx->flags;



    *data_size = 0;

   

   /* no supplementary picture */

    if (buf_size == 0) {

        return 0;

    }

    

    if(s->flags&CODEC_FLAG_TRUNCATED){

        int next;

        ParseContext *pc= &s->parse_context;

        

        pc->last_index= pc->index;



        if(s->codec_id==CODEC_ID_MPEG4){

            next= mpeg4_find_frame_end(s, buf, buf_size);

        }else{

            fprintf(stderr, "this codec doesnt support truncated bitstreams\n");

            return -1;

        }

        if(next==-1){

            if(buf_size + FF_INPUT_BUFFER_PADDING_SIZE + pc->index > pc->buffer_size){

                pc->buffer_size= buf_size + pc->index + 10*1024;

                pc->buffer= realloc(pc->buffer, pc->buffer_size);

            }



            memcpy(&pc->buffer[pc->index], buf, buf_size);

            pc->index += buf_size;

            return buf_size;

        }



        if(pc->index){

            if(next + FF_INPUT_BUFFER_PADDING_SIZE + pc->index > pc->buffer_size){

                pc->buffer_size= next + pc->index + 10*1024;

                pc->buffer= realloc(pc->buffer, pc->buffer_size);

            }



            memcpy(&pc->buffer[pc->index], buf, next + FF_INPUT_BUFFER_PADDING_SIZE );

            pc->index = 0;

            buf= pc->buffer;

            buf_size= pc->last_index + next;

        }

    }



retry:

    

    if(s->bitstream_buffer_size && buf_size<20){ //divx 5.01+ frame reorder

        init_get_bits(&s->gb, s->bitstream_buffer, s->bitstream_buffer_size);

    }else

        init_get_bits(&s->gb, buf, buf_size);

    s->bitstream_buffer_size=0;



    if (!s->context_initialized) {

        if (MPV_common_init(s) < 0) //we need the idct permutaton for reading a custom matrix

            return -1;

    }

      

    /* let's go :-) */

    if (s->msmpeg4_version==5) {

        ret= ff_wmv2_decode_picture_header(s);

    } else if (s->msmpeg4_version) {

        ret = msmpeg4_decode_picture_header(s);

    } else if (s->h263_pred) {

        if(s->avctx->extradata_size && s->picture_number==0){

            GetBitContext gb;

            

            init_get_bits(&gb, s->avctx->extradata, s->avctx->extradata_size);

            ret = ff_mpeg4_decode_picture_header(s, &gb);

        }

        ret = ff_mpeg4_decode_picture_header(s, &s->gb);



        if(s->flags& CODEC_FLAG_LOW_DELAY)

            s->low_delay=1;

    } else if (s->h263_intel) {

        ret = intel_h263_decode_picture_header(s);

    } else {

        ret = h263_decode_picture_header(s);

    }

    avctx->has_b_frames= !s->low_delay;



    if(s->workaround_bugs&FF_BUG_AUTODETECT){

        if(s->padding_bug_score > -2 && !s->data_partitioning)

            s->workaround_bugs |=  FF_BUG_NO_PADDING;

        else

            s->workaround_bugs &= ~FF_BUG_NO_PADDING;



        if(s->avctx->fourcc == ff_get_fourcc("XVIX")) 

            s->workaround_bugs|= FF_BUG_XVID_ILACE;

#if 0

        if(s->avctx->fourcc == ff_get_fourcc("MP4S")) 

            s->workaround_bugs|= FF_BUG_AC_VLC;

        

        if(s->avctx->fourcc == ff_get_fourcc("M4S2")) 

            s->workaround_bugs|= FF_BUG_AC_VLC;

#endif

        if(s->avctx->fourcc == ff_get_fourcc("UMP4")){

            s->workaround_bugs|= FF_BUG_UMP4;

            s->workaround_bugs|= FF_BUG_AC_VLC;

        }



        if(s->divx_version){

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA;

        }



        if(s->avctx->fourcc == ff_get_fourcc("XVID") && s->xvid_build==0)

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA;

        

        if(s->avctx->fourcc == ff_get_fourcc("XVID") && s->xvid_build==0)

            s->padding_bug_score= 256*256*256*64;

        

        if(s->xvid_build && s->xvid_build<=3)

            s->padding_bug_score= 256*256*256*64;

        

        if(s->xvid_build && s->xvid_build<=1)

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA;



#define SET_QPEL_FUNC(postfix1, postfix2) \

    s->dsp.put_ ## postfix1 = ff_put_ ## postfix2;\

    s->dsp.put_no_rnd_ ## postfix1 = ff_put_no_rnd_ ## postfix2;\

    s->dsp.avg_ ## postfix1 = ff_avg_ ## postfix2;



        if(s->lavc_build && s->lavc_build<4653)

            s->workaround_bugs|= FF_BUG_STD_QPEL;

        

//printf("padding_bug_score: %d\n", s->padding_bug_score);

#if 0

        if(s->divx_version==500)

            s->workaround_bugs|= FF_BUG_NO_PADDING;



        /* very ugly XVID padding bug detection FIXME/XXX solve this differently

         * lets hope this at least works

         */

        if(   s->resync_marker==0 && s->data_partitioning==0 && s->divx_version==0

           && s->codec_id==CODEC_ID_MPEG4 && s->vo_type==0)

            s->workaround_bugs|= FF_BUG_NO_PADDING;

        

        if(s->lavc_build && s->lavc_build<4609) //FIXME not sure about the version num but a 4609 file seems ok

            s->workaround_bugs|= FF_BUG_NO_PADDING;

#endif

    }

    

    if(s->workaround_bugs& FF_BUG_STD_QPEL){

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 5], qpel16_mc11_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 7], qpel16_mc31_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 9], qpel16_mc12_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][11], qpel16_mc32_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][13], qpel16_mc13_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][15], qpel16_mc33_old_c)



        SET_QPEL_FUNC(qpel_pixels_tab[1][ 5], qpel8_mc11_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][ 7], qpel8_mc31_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][ 9], qpel8_mc12_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][11], qpel8_mc32_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][13], qpel8_mc13_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][15], qpel8_mc33_old_c)

    }



#if 0 // dump bits per frame / qp / complexity

{

    static FILE *f=NULL;

    if(!f) f=fopen("rate_qp_cplx.txt", "w");

    fprintf(f, "%d %d %f\n", buf_size, s->qscale, buf_size*(double)s->qscale);

}

#endif

       

        /* After H263 & mpeg4 header decode we have the height, width,*/

        /* and other parameters. So then we could init the picture   */

        /* FIXME: By the way H263 decoder is evolving it should have */

        /* an H263EncContext                                         */

    if(s->aspected_height)

        new_aspect= s->aspected_width*s->width / (float)(s->height*s->aspected_height);

    else

        new_aspect=0;

    

    if (   s->width != avctx->width || s->height != avctx->height 

        || ABS(new_aspect - avctx->aspect_ratio) > 0.001) {

        /* H.263 could change picture size any time */

        MPV_common_end(s);

        s->context_initialized=0;

    }

    if (!s->context_initialized) {

        avctx->width = s->width;

        avctx->height = s->height;

        avctx->aspect_ratio= new_aspect;



        goto retry;

    }



    if((s->codec_id==CODEC_ID_H263 || s->codec_id==CODEC_ID_H263P))

        s->gob_index = ff_h263_get_gob_height(s);



    if(ret==FRAME_SKIPED) return get_consumed_bytes(s, buf_size);

    /* skip if the header was thrashed */

    if (ret < 0){

        fprintf(stderr, "header damaged\n");

        return -1;

    }

    

    // for hurry_up==5

    s->current_picture.pict_type= s->pict_type;

    s->current_picture.key_frame= s->pict_type == I_TYPE;



    /* skip b frames if we dont have reference frames */

    if(s->last_picture.data[0]==NULL && s->pict_type==B_TYPE) return get_consumed_bytes(s, buf_size);

    /* skip b frames if we are in a hurry */

    if(avctx->hurry_up && s->pict_type==B_TYPE) return get_consumed_bytes(s, buf_size);

    /* skip everything if we are in a hurry>=5 */

    if(avctx->hurry_up>=5) return get_consumed_bytes(s, buf_size);

    

    if(s->next_p_frame_damaged){

        if(s->pict_type==B_TYPE)

            return get_consumed_bytes(s, buf_size);

        else

            s->next_p_frame_damaged=0;

    }



    if(MPV_frame_start(s, avctx) < 0)

        return -1;



#ifdef DEBUG

    printf("qscale=%d\n", s->qscale);

#endif



    if(s->error_resilience)

        memset(s->error_status_table, MV_ERROR|AC_ERROR|DC_ERROR|VP_START|AC_END|DC_END|MV_END, s->mb_num*sizeof(UINT8));

    

    /* decode each macroblock */

    s->block_wrap[0]=

    s->block_wrap[1]=

    s->block_wrap[2]=

    s->block_wrap[3]= s->mb_width*2 + 2;

    s->block_wrap[4]=

    s->block_wrap[5]= s->mb_width + 2;

    s->mb_x=0; 

    s->mb_y=0;

    

    decode_slice(s);

    s->error_status_table[0]|= VP_START;

    while(s->mb_y<s->mb_height && s->gb.size*8 - get_bits_count(&s->gb)>16){

        if(s->msmpeg4_version){

            if(s->mb_x!=0 || (s->mb_y%s->slice_height)!=0)

                break;

        }else{

            if(ff_h263_resync(s)<0)

                break;

        }

        

        if(s->msmpeg4_version<4 && s->h263_pred)

            ff_mpeg4_clean_buffers(s);



        decode_slice(s);



        s->error_status_table[s->resync_mb_x + s->resync_mb_y*s->mb_width]|= VP_START;

    }



    if (s->h263_msmpeg4 && s->msmpeg4_version<4 && s->pict_type==I_TYPE)

        if(msmpeg4_decode_ext_header(s, buf_size) < 0){

            s->error_status_table[s->mb_num-1]= AC_ERROR|DC_ERROR|MV_ERROR;

        }

    

    /* divx 5.01+ bistream reorder stuff */

    if(s->codec_id==CODEC_ID_MPEG4 && s->bitstream_buffer_size==0 && s->divx_version>=500){

        int current_pos= get_bits_count(&s->gb)>>3;



        if(   buf_size - current_pos > 5 

           && buf_size - current_pos < BITSTREAM_BUFFER_SIZE){

            int i;

            int startcode_found=0;

            for(i=current_pos; i<buf_size-3; i++){

                if(buf[i]==0 && buf[i+1]==0 && buf[i+2]==1 && buf[i+3]==0xB6){

                    startcode_found=1;

                    break;

                }

            }

            if(startcode_found){

                memcpy(s->bitstream_buffer, buf + current_pos, buf_size - current_pos);

                s->bitstream_buffer_size= buf_size - current_pos;

            }

        }

    }



    if(s->error_resilience){

        int error=0, num_end_markers=0;

        for(i=0; i<s->mb_num; i++){

            int status= s->error_status_table[i];

#if 0

            if(i%s->mb_width == 0) printf("\n");

            printf("%2X ", status); 

#endif

            if(status==0) continue;



            if(status&(DC_ERROR|AC_ERROR|MV_ERROR))

                error=1;

            if(status&VP_START){

                if(num_end_markers) 

                    error=1;

                num_end_markers=3;

            }

            if(status&AC_END)

                num_end_markers--;

            if(status&DC_END)

                num_end_markers--;

            if(status&MV_END)

                num_end_markers--;

        }

        if(num_end_markers || error){

            fprintf(stderr, "concealing errors\n");

            ff_error_resilience(s);

        }

    }



    MPV_frame_end(s);



    if((avctx->debug&FF_DEBUG_VIS_MV) && s->last_picture.data[0]){

        const int shift= 1 + s->quarter_sample;

        int mb_y;

        uint8_t *ptr= s->last_picture.data[0];

        s->low_delay=0; //needed to see the vectors without trashing the buffers



        for(mb_y=0; mb_y<s->mb_height; mb_y++){

            int mb_x;

            for(mb_x=0; mb_x<s->mb_width; mb_x++){

                const int mb_index= mb_x + mb_y*s->mb_width;

                if(s->co_located_type_table[mb_index] == MV_TYPE_8X8){

                    int i;

                    for(i=0; i<4; i++){

                        int sx= mb_x*16 + 4 + 8*(i&1);

                        int sy= mb_y*16 + 4 + 8*(i>>1);

                        int xy= 1 + mb_x*2 + (i&1) + (mb_y*2 + 1 + (i>>1))*(s->mb_width*2 + 2);

                        int mx= (s->motion_val[xy][0]>>shift) + sx;

                        int my= (s->motion_val[xy][1]>>shift) + sy;

                        draw_line(ptr, sx, sy, mx, my, s->width, s->height, s->linesize, 100);

                    }

                }else{

                    int sx= mb_x*16 + 8;

                    int sy= mb_y*16 + 8;

                    int xy= 1 + mb_x*2 + (mb_y*2 + 1)*(s->mb_width*2 + 2);

                    int mx= (s->motion_val[xy][0]>>shift) + sx;

                    int my= (s->motion_val[xy][1]>>shift) + sy;

                    draw_line(ptr, sx, sy, mx, my, s->width, s->height, s->linesize, 100);

                }

                s->mbskip_table[mb_index]=0;

            }

        }

    }





    if(s->pict_type==B_TYPE || s->low_delay){

        *pict= *(AVFrame*)&s->current_picture;

    } else {

        *pict= *(AVFrame*)&s->last_picture;

    }



    if(avctx->debug&FF_DEBUG_QP){

        int8_t *qtab= pict->qscale_table;

        int x,y;

        

        for(y=0; y<s->mb_height; y++){

            for(x=0; x<s->mb_width; x++){

                printf("%2d ", qtab[x + y*s->mb_width]);

            }

            printf("\n");

        }

        printf("\n");

    }



    /* Return the Picture timestamp as the frame number */

    /* we substract 1 because it is added on utils.c    */

    avctx->frame_number = s->picture_number - 1;



    /* dont output the last pic after seeking */

    if(s->last_picture.data[0] || s->low_delay)

        *data_size = sizeof(AVFrame);

#ifdef PRINT_FRAME_TIME

printf("%Ld\n", rdtsc()-time);

#endif

    return get_consumed_bytes(s, buf_size);

}
