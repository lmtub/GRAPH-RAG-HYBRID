static av_cold int flashsv_encode_init(AVCodecContext *avctx)

{

    FlashSVContext *s = avctx->priv_data;



    s->avctx = avctx;



    if (avctx->width > 4095 || avctx->height > 4095) {

        av_log(avctx, AV_LOG_ERROR,

               "Input dimensions too large, input must be max 4096x4096 !\n");

        return AVERROR_INVALIDDATA;

    }



    // Needed if zlib unused or init aborted before deflateInit

    memset(&s->zstream, 0, sizeof(z_stream));



    s->last_key_frame = 0;



    s->image_width  = avctx->width;

    s->image_height = avctx->height;



    s->tmpblock  = av_mallocz(3 * 256 * 256);

    s->encbuffer = av_mallocz(s->image_width * s->image_height * 3);



    if (!s->tmpblock || !s->encbuffer) {

        av_log(avctx, AV_LOG_ERROR, "Memory allocation failed.\n");

        return AVERROR(ENOMEM);

    }



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame) {

        flashsv_encode_end(avctx);

        return AVERROR(ENOMEM);

    }



    return 0;

}
