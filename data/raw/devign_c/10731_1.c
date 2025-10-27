static int encode_init(AVCodecContext *avctx)

{

    HYuvContext *s = avctx->priv_data;

    int i, j, width, height;



    s->avctx= avctx;

    s->flags= avctx->flags;

        

    dsputil_init(&s->dsp, avctx);

    

    width= s->width= avctx->width;

    height= s->height= avctx->height;

    

    assert(width && height);

    

    avctx->extradata= av_mallocz(1024*30);

    avctx->stats_out= av_mallocz(1024*30);

    s->version=2;

    

    avctx->coded_frame= &s->picture;

    

    switch(avctx->pix_fmt){

    case PIX_FMT_YUV420P:

        s->bitstream_bpp= 12;

        break;

    case PIX_FMT_YUV422P:

        s->bitstream_bpp= 16;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "format not supported\n");

        return -1;

    }

    avctx->bits_per_sample= s->bitstream_bpp;

    s->decorrelate= s->bitstream_bpp >= 24;

    s->predictor= avctx->prediction_method;

    s->interlaced= avctx->flags&CODEC_FLAG_INTERLACED_ME ? 1 : 0;

    if(avctx->context_model==1){

        s->context= avctx->context_model;

        if(s->flags & (CODEC_FLAG_PASS1|CODEC_FLAG_PASS2)){

            av_log(avctx, AV_LOG_ERROR, "context=1 is not compatible with 2 pass huffyuv encoding\n");

            return -1;

        }

    }else s->context= 0;

    

    if(avctx->codec->id==CODEC_ID_HUFFYUV){

        if(avctx->pix_fmt==PIX_FMT_YUV420P){

            av_log(avctx, AV_LOG_ERROR, "Error: YV12 is not supported by huffyuv; use vcodec=ffvhuff or format=422p\n");

            return -1;

        }

        if(avctx->context_model){

            av_log(avctx, AV_LOG_ERROR, "Error: per-frame huffman tables are not supported by huffyuv; use vcodec=ffvhuff\n");

            return -1;

        }

        if(s->interlaced != ( height > 288 ))

            av_log(avctx, AV_LOG_INFO, "using huffyuv 2.2.0 or newer interlacing flag\n");

    }else if(avctx->strict_std_compliance>=0){

        av_log(avctx, AV_LOG_ERROR, "This codec is under development; files encoded with it may not be decodeable with future versions!!! Set vstrict=-1 to use it anyway.\n");

        return -1;

    }

    

    ((uint8_t*)avctx->extradata)[0]= s->predictor;

    ((uint8_t*)avctx->extradata)[1]= s->bitstream_bpp;

    ((uint8_t*)avctx->extradata)[2]= s->interlaced ? 0x10 : 0x20;

    if(s->context)

        ((uint8_t*)avctx->extradata)[2]|= 0x40;

    ((uint8_t*)avctx->extradata)[3]= 0;

    s->avctx->extradata_size= 4;

    

    if(avctx->stats_in){

        char *p= avctx->stats_in;

    

        for(i=0; i<3; i++)

            for(j=0; j<256; j++)

                s->stats[i][j]= 1;



        for(;;){

            for(i=0; i<3; i++){

                char *next;



                for(j=0; j<256; j++){

                    s->stats[i][j]+= strtol(p, &next, 0);

                    if(next==p) return -1;

                    p=next;

                }        

            }

            if(p[0]==0 || p[1]==0 || p[2]==0) break;

        }

    }else{

        for(i=0; i<3; i++)

            for(j=0; j<256; j++){

                int d= FFMIN(j, 256-j);

                

                s->stats[i][j]= 100000000/(d+1);

            }

    }

    

    for(i=0; i<3; i++){

        generate_len_table(s->len[i], s->stats[i], 256);



        if(generate_bits_table(s->bits[i], s->len[i])<0){

            return -1;

        }

        

        s->avctx->extradata_size+=

        store_table(s, s->len[i], &((uint8_t*)s->avctx->extradata)[s->avctx->extradata_size]);

    }



    if(s->context){

        for(i=0; i<3; i++){

            int pels = width*height / (i?40:10);

            for(j=0; j<256; j++){

                int d= FFMIN(j, 256-j);

                s->stats[i][j]= pels/(d+1);

            }

        }

    }else{

        for(i=0; i<3; i++)

            for(j=0; j<256; j++)

                s->stats[i][j]= 0;

    }

    

//    printf("pred:%d bpp:%d hbpp:%d il:%d\n", s->predictor, s->bitstream_bpp, avctx->bits_per_sample, s->interlaced);



    s->picture_number=0;



    return 0;

}
