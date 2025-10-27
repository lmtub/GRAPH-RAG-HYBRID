static av_cold int m101_decode_init(AVCodecContext *avctx)

{

    if (avctx->extradata_size < 6*4)

        return AVERROR_INVALIDDATA;



    if (avctx->extradata[2*4] == 10)

        avctx->pix_fmt = AV_PIX_FMT_YUV422P10;

    else

        avctx->pix_fmt = AV_PIX_FMT_YUYV422;





    return 0;

}
