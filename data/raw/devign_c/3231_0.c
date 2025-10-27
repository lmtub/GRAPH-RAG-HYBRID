static av_cold int vaapi_encode_h264_init(AVCodecContext *avctx)

{

    return ff_vaapi_encode_init(avctx, &vaapi_encode_type_h264);

}
