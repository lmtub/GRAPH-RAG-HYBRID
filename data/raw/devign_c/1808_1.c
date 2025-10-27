static av_cold int smvjpeg_decode_init(AVCodecContext *avctx)

{

    SMVJpegDecodeContext *s = avctx->priv_data;

    AVCodec *codec;

    AVDictionary *thread_opt = NULL;

    int ret = 0;



    s->frames_per_jpeg = 0;



    s->picture[0] = av_frame_alloc();

    if (!s->picture[0])

        return AVERROR(ENOMEM);



    s->picture[1] = av_frame_alloc();

    if (!s->picture[1])

        return AVERROR(ENOMEM);



    s->jpg.picture_ptr      = s->picture[0];



    if (avctx->extradata_size >= 4)

        s->frames_per_jpeg = AV_RL32(avctx->extradata);



    if (s->frames_per_jpeg <= 0) {

        av_log(avctx, AV_LOG_ERROR, "Invalid number of frames per jpeg.\n");

        ret = -1;

    }



    codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);

    if (!codec) {

        av_log(avctx, AV_LOG_ERROR, "MJPEG codec not found\n");

        ret = -1;

    }



    s->avctx = avcodec_alloc_context3(codec);



    av_dict_set(&thread_opt, "threads", "1", 0);

    s->avctx->refcounted_frames = 1;

    s->avctx->flags = avctx->flags;

    s->avctx->idct_algo = avctx->idct_algo;

    if (ff_codec_open2_recursive(s->avctx, codec, &thread_opt) < 0) {

        av_log(avctx, AV_LOG_ERROR, "MJPEG codec failed to open\n");

        ret = -1;

    }

    av_dict_free(&thread_opt);



    return ret;

}
