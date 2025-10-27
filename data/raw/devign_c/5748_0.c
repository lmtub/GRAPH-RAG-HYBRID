static int has_codec_parameters(AVCodecContext *enc)

{

    int val;

    switch(enc->codec_type) {

    case CODEC_TYPE_AUDIO:

        val = enc->sample_rate;

        break;

    case CODEC_TYPE_VIDEO:

        val = enc->width && enc->pix_fmt != PIX_FMT_NONE;

        break;

    default:

        val = 1;

        break;

    }

    return (val != 0);

}
