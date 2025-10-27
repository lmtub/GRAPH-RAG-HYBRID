static av_cold int alac_decode_close(AVCodecContext *avctx)

{

    ALACContext *alac = avctx->priv_data;



    int chan;

    for (chan = 0; chan < alac->numchannels; chan++) {

        av_freep(&alac->predicterror_buffer[chan]);

        av_freep(&alac->outputsamples_buffer[chan]);

        av_freep(&alac->wasted_bits_buffer[chan]);

    }



    return 0;

}
