static av_cold int ffat_close_decoder(AVCodecContext *avctx)

{

    ATDecodeContext *at = avctx->priv_data;

    if (at->converter)

        AudioConverterDispose(at->converter);

    av_packet_unref(&at->new_in_pkt);

    av_packet_unref(&at->in_pkt);

    av_free(at->decoded_data);

    av_free(at->extradata);

    return 0;

}
