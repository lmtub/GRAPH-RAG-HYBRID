static void add_codec(FFStream *stream, AVCodecContext *av)

{

    AVStream *st;



    if(stream->nb_streams >= FF_ARRAY_ELEMS(stream->streams))

        return;



    /* compute default parameters */

    switch(av->codec_type) {

    case AVMEDIA_TYPE_AUDIO:

        if (av->bit_rate == 0)

            av->bit_rate = 64000;

        if (av->sample_rate == 0)

            av->sample_rate = 22050;

        if (av->channels == 0)

            av->channels = 1;

        break;

    case AVMEDIA_TYPE_VIDEO:

        if (av->bit_rate == 0)

            av->bit_rate = 64000;

        if (av->time_base.num == 0){

            av->time_base.den = 5;

            av->time_base.num = 1;

        }

        if (av->width == 0 || av->height == 0) {

            av->width = 160;

            av->height = 128;

        }

        /* Bitrate tolerance is less for streaming */

        if (av->bit_rate_tolerance == 0)

            av->bit_rate_tolerance = FFMAX(av->bit_rate / 4,

                      (int64_t)av->bit_rate*av->time_base.num/av->time_base.den);

        if (av->qmin == 0)

            av->qmin = 3;

        if (av->qmax == 0)

            av->qmax = 31;

        if (av->max_qdiff == 0)

            av->max_qdiff = 3;

        av->qcompress = 0.5;

        av->qblur = 0.5;



        if (!av->nsse_weight)

            av->nsse_weight = 8;



        av->frame_skip_cmp = FF_CMP_DCTMAX;

        if (!av->me_method)

            av->me_method = ME_EPZS;

        av->rc_buffer_aggressivity = 1.0;



        if (!av->rc_eq)

            av->rc_eq = "tex^qComp";

        if (!av->i_quant_factor)

            av->i_quant_factor = -0.8;

        if (!av->b_quant_factor)

            av->b_quant_factor = 1.25;

        if (!av->b_quant_offset)

            av->b_quant_offset = 1.25;

        if (!av->rc_max_rate)

            av->rc_max_rate = av->bit_rate * 2;



        if (av->rc_max_rate && !av->rc_buffer_size) {

            av->rc_buffer_size = av->rc_max_rate;

        }





        break;

    default:

        abort();

    }



    st = av_mallocz(sizeof(AVStream));

    if (!st)

        return;

    st->codec = avcodec_alloc_context3(NULL);

    stream->streams[stream->nb_streams++] = st;

    memcpy(st->codec, av, sizeof(AVCodecContext));

}
