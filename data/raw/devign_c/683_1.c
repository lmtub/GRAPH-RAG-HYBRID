static int encode_frame(AVCodecContext *avctx, unsigned char *buf, int buf_size, void *data){

    FFV1Context *f = avctx->priv_data;

    CABACContext * const c= &f->c;

    AVFrame *pict = data;

    const int width= f->width;

    const int height= f->height;

    AVFrame * const p= &f->picture;

    int used_count= 0;



    if(avctx->strict_std_compliance >= 0){

        av_log(avctx, AV_LOG_ERROR, "this codec is under development, files encoded with it wont be decodeable with future versions!!!\n"

               "use vstrict=-1 to use it anyway\n");

        return -1;

    }

        

    ff_init_cabac_encoder(c, buf, buf_size);

    ff_init_cabac_states(c, ff_h264_lps_range, ff_h264_mps_state, ff_h264_lps_state, 64);

    c->lps_state[2] = 1;

    c->lps_state[3] = 0;

    

    *p = *pict;

    p->pict_type= FF_I_TYPE;

    

    if(avctx->gop_size==0 || f->picture_number % avctx->gop_size == 0){

        put_cabac_bypass(c, 1);

        p->key_frame= 1;

        write_header(f);

        clear_state(f);

    }else{

        put_cabac_bypass(c, 0);

        p->key_frame= 0;

    }



    if(!f->ac){

        used_count += put_cabac_terminate(c, 1);

//printf("pos=%d\n", used_count);

        init_put_bits(&f->pb, buf + used_count, buf_size - used_count);

    }

    

    if(f->colorspace==0){

        const int chroma_width = -((-width )>>f->chroma_h_shift);

        const int chroma_height= -((-height)>>f->chroma_v_shift);



        encode_plane(f, p->data[0], width, height, p->linesize[0], 0);



        encode_plane(f, p->data[1], chroma_width, chroma_height, p->linesize[1], 1);

        encode_plane(f, p->data[2], chroma_width, chroma_height, p->linesize[2], 1);

    }else{

        encode_rgb_frame(f, (uint32_t*)(p->data[0]), width, height, p->linesize[0]/4);

    }

    emms_c();

    

    f->picture_number++;



    if(f->ac){

        return put_cabac_terminate(c, 1);

    }else{

        flush_put_bits(&f->pb); //nicer padding FIXME

        return used_count + (put_bits_count(&f->pb)+7)/8;

    }

}
