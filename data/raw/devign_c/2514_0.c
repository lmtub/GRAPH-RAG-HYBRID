static av_cold int rv10_decode_init(AVCodecContext *avctx)

{

    RVDecContext  *rv = avctx->priv_data;

    MpegEncContext *s = &rv->m;

    static int done=0;

    int major_ver, minor_ver, micro_ver;



    if (avctx->extradata_size < 8) {

        av_log(avctx, AV_LOG_ERROR, "Extradata is too small.\n");

        return -1;

    }



    ff_MPV_decode_defaults(s);



    s->avctx= avctx;

    s->out_format = FMT_H263;

    s->codec_id= avctx->codec_id;



    s->orig_width = s->width  = avctx->coded_width;

    s->orig_height= s->height = avctx->coded_height;



    s->h263_long_vectors= ((uint8_t*)avctx->extradata)[3] & 1;

    rv->sub_id = AV_RB32((uint8_t*)avctx->extradata + 4);



    major_ver = RV_GET_MAJOR_VER(rv->sub_id);

    minor_ver = RV_GET_MINOR_VER(rv->sub_id);

    micro_ver = RV_GET_MICRO_VER(rv->sub_id);



    s->low_delay = 1;

    switch (major_ver) {

    case 1:

        s->rv10_version = micro_ver ? 3 : 1;

        s->obmc = micro_ver == 2;

        break;

    case 2:

        if (minor_ver >= 2) {

            s->low_delay = 0;

            s->avctx->has_b_frames = 1;

        }

        break;

    default:

        av_log(s->avctx, AV_LOG_ERROR, "unknown header %X\n", rv->sub_id);

        av_log_missing_feature(avctx, "RV1/2 version", 1);

        return AVERROR_PATCHWELCOME;

    }



    if(avctx->debug & FF_DEBUG_PICT_INFO){

        av_log(avctx, AV_LOG_DEBUG, "ver:%X ver0:%X\n", rv->sub_id, avctx->extradata_size >= 4 ? ((uint32_t*)avctx->extradata)[0] : -1);

    }



    avctx->pix_fmt = AV_PIX_FMT_YUV420P;



    if (ff_MPV_common_init(s) < 0)

        return -1;



    ff_h263_decode_init_vlc();



    /* init rv vlc */

    if (!done) {

        INIT_VLC_STATIC(&rv_dc_lum, DC_VLC_BITS, 256,

                 rv_lum_bits, 1, 1,

                 rv_lum_code, 2, 2, 16384);

        INIT_VLC_STATIC(&rv_dc_chrom, DC_VLC_BITS, 256,

                 rv_chrom_bits, 1, 1,

                 rv_chrom_code, 2, 2, 16388);

        done = 1;

    }



    return 0;

}
