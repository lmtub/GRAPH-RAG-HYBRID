void add_codec(FFStream *stream, AVCodecContext *av)

{

    AVStream *st;



    /* compute default parameters */

    switch(av->codec_type) {

    case CODEC_TYPE_AUDIO:

        if (av->bit_rate == 0)

            av->bit_rate = 64000;

        if (av->sample_rate == 0)

            av->sample_rate = 22050;

        if (av->channels == 0)

            av->channels = 1;

        break;

    case CODEC_TYPE_VIDEO:

        if (av->bit_rate == 0)

            av->bit_rate = 64000;

        if (av->frame_rate == 0)

            av->frame_rate = 5 * FRAME_RATE_BASE;

        if (av->width == 0 || av->height == 0) {

            av->width = 160;

            av->height = 128;

        }

        /* Bitrate tolerance is less for streaming */

        if (av->bit_rate_tolerance == 0)

            av->bit_rate_tolerance = av->bit_rate / 4;

        if (av->qmin == 0)

            av->qmin = 3;

        if (av->qmax == 0)

            av->qmax = 31;

        if (av->max_qdiff == 0)

            av->max_qdiff = 3;

        av->qcompress = 0.5;

        av->qblur = 0.5;



        break;

    default:

        abort();

    }



    st = av_mallocz(sizeof(AVStream));

    if (!st)

        return;

    stream->streams[stream->nb_streams++] = st;

    memcpy(&st->codec, av, sizeof(AVCodecContext));

}
