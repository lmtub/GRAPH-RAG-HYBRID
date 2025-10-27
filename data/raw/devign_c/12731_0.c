static av_cold int raw_init_decoder(AVCodecContext *avctx)

{

    RawVideoContext *context = avctx->priv_data;

    const AVPixFmtDescriptor *desc;



    ff_bswapdsp_init(&context->bbdsp);



    if (   avctx->codec_tag == MKTAG('r','a','w',' ')

        || avctx->codec_tag == MKTAG('N','O','1','6'))

        avctx->pix_fmt = avpriv_find_pix_fmt(avpriv_pix_fmt_bps_mov,

                                      avctx->bits_per_coded_sample);

    else if (avctx->codec_tag == MKTAG('W', 'R', 'A', 'W'))

        avctx->pix_fmt = avpriv_find_pix_fmt(avpriv_pix_fmt_bps_avi,

                                      avctx->bits_per_coded_sample);

    else if (avctx->codec_tag && (avctx->codec_tag & 0xFFFFFF) != MKTAG('B','I','T', 0))

        avctx->pix_fmt = avpriv_find_pix_fmt(ff_raw_pix_fmt_tags, avctx->codec_tag);

    else if (avctx->pix_fmt == AV_PIX_FMT_NONE && avctx->bits_per_coded_sample)

        avctx->pix_fmt = avpriv_find_pix_fmt(avpriv_pix_fmt_bps_avi,

                                      avctx->bits_per_coded_sample);



    desc = av_pix_fmt_desc_get(avctx->pix_fmt);

    if (!desc) {

        av_log(avctx, AV_LOG_ERROR, "Invalid pixel format.\n");

        return AVERROR(EINVAL);

    }



    if (desc->flags & (AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_PSEUDOPAL)) {

        context->palette = av_buffer_alloc(AVPALETTE_SIZE);

        if (!context->palette)

            return AVERROR(ENOMEM);

        if (desc->flags & AV_PIX_FMT_FLAG_PSEUDOPAL)

            avpriv_set_systematic_pal2((uint32_t*)context->palette->data, avctx->pix_fmt);

        else

            memset(context->palette->data, 0, AVPALETTE_SIZE);

    }



    if ((avctx->extradata_size >= 9 &&

         !memcmp(avctx->extradata + avctx->extradata_size - 9, "BottomUp", 9)) ||

        avctx->codec_tag == MKTAG('c','y','u','v') ||

        avctx->codec_tag == MKTAG(3, 0, 0, 0) ||

        avctx->codec_tag == MKTAG('W','R','A','W'))

        context->flip = 1;



    if (avctx->codec_tag == AV_RL32("yuv2") &&

        avctx->pix_fmt   == AV_PIX_FMT_YUYV422)

        context->is_yuv2 = 1;



    return 0;

}
