static av_cold int pcx_encode_init(AVCodecContext *avctx)

{

    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);



    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;



    return 0;

}
