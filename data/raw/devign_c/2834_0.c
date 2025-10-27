int ff_mpa_decode_header(AVCodecContext *avctx, uint32_t head, int *sample_rate)

{

    MPADecodeContext s1, *s = &s1;

    s1.avctx = avctx;



    if (ff_mpa_check_header(head) != 0)

        return -1;



    if (ff_mpegaudio_decode_header(s, head) != 0) {

        return -1;

    }



    switch(s->layer) {

    case 1:

        avctx->frame_size = 384;

        break;

    case 2:

        avctx->frame_size = 1152;

        break;

    default:

    case 3:

        if (s->lsf)

            avctx->frame_size = 576;

        else

            avctx->frame_size = 1152;

        break;

    }



    *sample_rate = s->sample_rate;

    avctx->channels = s->nb_channels;

    avctx->bit_rate = s->bit_rate;

    avctx->sub_id = s->layer;

    return s->frame_size;

}
