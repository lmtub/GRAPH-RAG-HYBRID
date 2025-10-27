static av_cold int g726_encode_init(AVCodecContext *avctx)

{

    G726Context* c = avctx->priv_data;



    if (avctx->strict_std_compliance > FF_COMPLIANCE_UNOFFICIAL &&

        avctx->sample_rate != 8000) {

        av_log(avctx, AV_LOG_ERROR, "Sample rates other than 8kHz are not "

               "allowed when the compliance level is higher than unofficial. "

               "Resample or reduce the compliance level.\n");

        return AVERROR(EINVAL);

    }

    if (avctx->sample_rate <= 0) {

        av_log(avctx, AV_LOG_ERROR, "Samplerate is invalid\n");

        return -1;

    }



    if(avctx->channels != 1){

        av_log(avctx, AV_LOG_ERROR, "Only mono is supported\n");

        return -1;

    }



    if (avctx->bit_rate % avctx->sample_rate) {

        av_log(avctx, AV_LOG_ERROR, "Bitrate - Samplerate combination is invalid\n");

        return AVERROR(EINVAL);

    }

    c->code_size = (avctx->bit_rate + avctx->sample_rate/2) / avctx->sample_rate;

    if (c->code_size < 2 || c->code_size > 5) {

        av_log(avctx, AV_LOG_ERROR, "Invalid number of bits %d\n", c->code_size);

        return AVERROR(EINVAL);

    }

    avctx->bits_per_coded_sample = c->code_size;



    g726_reset(c, c->code_size - 2);



    avctx->coded_frame = avcodec_alloc_frame();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);

    avctx->coded_frame->key_frame = 1;



    /* select a frame size that will end on a byte boundary and have a size of

       approximately 1024 bytes */

    avctx->frame_size = ((int[]){ 4096, 2736, 2048, 1640 })[c->code_size - 2];



    return 0;

}
