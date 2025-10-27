static void compute_frame_duration(int *pnum, int *pden,

                                   AVFormatContext *s, AVStream *st, 

                                   AVCodecParserContext *pc, AVPacket *pkt)

{

    int frame_size;



    *pnum = 0;

    *pden = 0;

    switch(st->codec.codec_type) {

    case CODEC_TYPE_VIDEO:

        *pnum = st->codec.frame_rate_base;

        *pden = st->codec.frame_rate;

        if (pc && pc->repeat_pict) {

            *pden *= 2;

            *pnum = (*pnum) * (2 + pc->repeat_pict);

        }

        break;

    case CODEC_TYPE_AUDIO:

        frame_size = get_audio_frame_size(&st->codec, pkt->size);

        if (frame_size < 0)

            break;

        *pnum = frame_size;

        *pden = st->codec.sample_rate;

        break;

    default:

        break;

    }

}
