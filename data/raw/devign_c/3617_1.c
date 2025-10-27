static int select_reference_stream(AVFormatContext *s)

{

    SegmentContext *seg = s->priv_data;

    int ret, i;



    seg->reference_stream_index = -1;

    if (!strcmp(seg->reference_stream_specifier, "auto")) {

        /* select first index of type with highest priority */

        int type_index_map[AVMEDIA_TYPE_NB];

        static const enum AVMediaType type_priority_list[] = {

            AVMEDIA_TYPE_VIDEO,

            AVMEDIA_TYPE_AUDIO,

            AVMEDIA_TYPE_SUBTITLE,

            AVMEDIA_TYPE_DATA,

            AVMEDIA_TYPE_ATTACHMENT

        };

        enum AVMediaType type;



        for (i = 0; i < AVMEDIA_TYPE_NB; i++)

            type_index_map[i] = -1;



        /* select first index for each type */

        for (i = 0; i < s->nb_streams; i++) {

            type = s->streams[i]->codec->codec_type;

            if ((unsigned)type < AVMEDIA_TYPE_NB && type_index_map[type] == -1

                /* ignore attached pictures/cover art streams */

                && !(s->streams[i]->disposition & AV_DISPOSITION_ATTACHED_PIC))

                type_index_map[type] = i;

        }



        for (i = 0; i < FF_ARRAY_ELEMS(type_priority_list); i++) {

            type = type_priority_list[i];

            if ((seg->reference_stream_index = type_index_map[type]) >= 0)

                break;

        }

    } else {

        for (i = 0; i < s->nb_streams; i++) {

            ret = avformat_match_stream_specifier(s, s->streams[i],

                                                  seg->reference_stream_specifier);

            if (ret < 0)

                break;

            if (ret > 0) {

                seg->reference_stream_index = i;

                break;

            }

        }

    }



    if (seg->reference_stream_index < 0) {

        av_log(s, AV_LOG_ERROR, "Could not select stream matching identifier '%s'\n",

               seg->reference_stream_specifier);

        return AVERROR(EINVAL);

    }



    return 0;

}
