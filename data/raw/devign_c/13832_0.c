static av_cold void ffat_encode_flush(AVCodecContext *avctx)

{

    ATDecodeContext *at = avctx->priv_data;

    AudioConverterReset(at->converter);

    av_frame_unref(&at->new_in_frame);

    av_frame_unref(&at->in_frame);

}
