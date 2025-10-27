static av_cold int dvvideo_encode_init(AVCodecContext *avctx)

{

    DVVideoContext *s = avctx->priv_data;

    FDCTDSPContext fdsp;

    MECmpContext mecc;

    PixblockDSPContext pdsp;

    int ret;



    s->sys = av_dv_codec_profile(avctx->width, avctx->height, avctx->pix_fmt);

    if (!s->sys) {

        av_log(avctx, AV_LOG_ERROR, "Found no DV profile for %ix%i %s video. "

                                    "Valid DV profiles are:\n",

               avctx->width, avctx->height, av_get_pix_fmt_name(avctx->pix_fmt));

        ff_dv_print_profiles(avctx, AV_LOG_ERROR);

        return AVERROR(EINVAL);

    }

    ret = ff_dv_init_dynamic_tables(s, s->sys);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error initializing work tables.\n");

        return ret;

    }



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);



    dv_vlc_map_tableinit();



    ff_fdctdsp_init(&fdsp, avctx);

    ff_me_cmp_init(&mecc, avctx);

    ff_pixblockdsp_init(&pdsp, avctx);

    ff_set_cmp(&mecc, mecc.ildct_cmp, avctx->ildct_cmp);



    s->get_pixels = pdsp.get_pixels;

    s->ildct_cmp  = mecc.ildct_cmp[5];



    s->fdct[0]    = fdsp.fdct;

    s->fdct[1]    = fdsp.fdct248;



    return ff_dvvideo_init(avctx);

}
