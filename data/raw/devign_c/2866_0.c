static av_cold int png_enc_init(AVCodecContext *avctx)

{

    PNGEncContext *s = avctx->priv_data;



#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    ff_huffyuvencdsp_init(&s->hdsp);



    s->filter_type = av_clip(avctx->prediction_method,

                             PNG_FILTER_VALUE_NONE,

                             PNG_FILTER_VALUE_MIXED);

    if (avctx->pix_fmt == AV_PIX_FMT_MONOBLACK)

        s->filter_type = PNG_FILTER_VALUE_NONE;



    return 0;

}
