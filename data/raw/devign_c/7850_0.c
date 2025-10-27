static int config(struct vf_instance *vf,

        int width, int height, int d_width, int d_height,

        unsigned int flags, unsigned int outfmt){

        int i;

        AVCodec *enc= avcodec_find_encoder(AV_CODEC_ID_SNOW);



        for(i=0; i<3; i++){

            int is_chroma= !!i;

            int w= ((width  + 4*BLOCK-1) & (~(2*BLOCK-1)))>>is_chroma;

            int h= ((height + 4*BLOCK-1) & (~(2*BLOCK-1)))>>is_chroma;



            vf->priv->temp_stride[i]= w;

            vf->priv->temp[i]= malloc(vf->priv->temp_stride[i]*h*sizeof(int16_t));

            vf->priv->src [i]= malloc(vf->priv->temp_stride[i]*h*sizeof(uint8_t));

        }

        for(i=0; i< (1<<vf->priv->log2_count); i++){

            AVCodecContext *avctx_enc;

            AVDictionary *opts = NULL;



            avctx_enc=

            vf->priv->avctx_enc[i]= avcodec_alloc_context3(NULL);

            avctx_enc->width = width + BLOCK;

            avctx_enc->height = height + BLOCK;

            avctx_enc->time_base= (AVRational){1,25};  // meaningless

            avctx_enc->gop_size = 300;

            avctx_enc->max_b_frames= 0;

            avctx_enc->pix_fmt = AV_PIX_FMT_YUV420P;

            avctx_enc->flags = CODEC_FLAG_QSCALE | CODEC_FLAG_LOW_DELAY;

            avctx_enc->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

            avctx_enc->global_quality= 123;

            av_dict_set(&opts, "no_bitstream", "1", 0);

            avcodec_open2(avctx_enc, enc, &opts);

            av_dict_free(&opts);

            assert(avctx_enc->codec);

        }

        vf->priv->frame= av_frame_alloc();

        vf->priv->frame_dec= av_frame_alloc();



        vf->priv->outbuf_size= (width + BLOCK)*(height + BLOCK)*10;

        vf->priv->outbuf= malloc(vf->priv->outbuf_size);



        return ff_vf_next_config(vf,width,height,d_width,d_height,flags,outfmt);

}
