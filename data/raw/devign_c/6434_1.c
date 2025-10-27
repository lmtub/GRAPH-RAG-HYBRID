int check_stream_specifier(AVFormatContext *s, AVStream *st, const char *spec)

{

    if (*spec <= '9' && *spec >= '0') /* opt:index */

        return strtol(spec, NULL, 0) == st->index;

    else if (*spec == 'v' || *spec == 'a' || *spec == 's' || *spec == 'd' ||

             *spec == 't') { /* opt:[vasdt] */

        enum AVMediaType type;



        switch (*spec++) {

        case 'v': type = AVMEDIA_TYPE_VIDEO;      break;

        case 'a': type = AVMEDIA_TYPE_AUDIO;      break;

        case 's': type = AVMEDIA_TYPE_SUBTITLE;   break;

        case 'd': type = AVMEDIA_TYPE_DATA;       break;

        case 't': type = AVMEDIA_TYPE_ATTACHMENT; break;


        }

        if (type != st->codec->codec_type)

            return 0;

        if (*spec++ == ':') { /* possibly followed by :index */

            int i, index = strtol(spec, NULL, 0);

            for (i = 0; i < s->nb_streams; i++)

                if (s->streams[i]->codec->codec_type == type && index-- == 0)

                   return i == st->index;

            return 0;

        }

        return 1;

    } else if (*spec == 'p' && *(spec + 1) == ':') {

        int prog_id, i, j;

        char *endptr;

        spec += 2;

        prog_id = strtol(spec, &endptr, 0);

        for (i = 0; i < s->nb_programs; i++) {

            if (s->programs[i]->id != prog_id)

                continue;



            if (*endptr++ == ':') {

                int stream_idx = strtol(endptr, NULL, 0);

                return stream_idx >= 0 &&

                    stream_idx < s->programs[i]->nb_stream_indexes &&

                    st->index == s->programs[i]->stream_index[stream_idx];

            }



            for (j = 0; j < s->programs[i]->nb_stream_indexes; j++)

                if (st->index == s->programs[i]->stream_index[j])

                    return 1;

        }

        return 0;

    } else if (!*spec) /* empty specifier, matches everything */

        return 1;



    av_log(s, AV_LOG_ERROR, "Invalid stream specifier: %s.\n", spec);

    return AVERROR(EINVAL);

}