av_cold int ff_nvenc_encode_init(AVCodecContext *avctx)

{

    int ret;



    if ((ret = nvenc_load_libraries(avctx)) < 0)

        return ret;



    if ((ret = nvenc_setup_device(avctx)) < 0)

        return ret;



    if ((ret = nvenc_setup_encoder(avctx)) < 0)

        return ret;



    if ((ret = nvenc_setup_surfaces(avctx)) < 0)

        return ret;



    if (avctx->flags & CODEC_FLAG_GLOBAL_HEADER) {

        if ((ret = nvenc_setup_extradata(avctx)) < 0)

            return ret;

    }



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);



    return 0;

}
