static int parse_adaptation_sets(AVFormatContext *s)

{

    WebMDashMuxContext *w = s->priv_data;

    char *p = w->adaptation_sets;

    char *q;

    enum { new_set, parsed_id, parsing_streams } state;

    if (!w->adaptation_sets) {

        av_log(s, AV_LOG_ERROR, "The 'adaptation_sets' option must be set.\n");

        return AVERROR(EINVAL);

    }

    // syntax id=0,streams=0,1,2 id=1,streams=3,4 and so on

    state = new_set;

    while (p < w->adaptation_sets + strlen(w->adaptation_sets)) {

        if (*p == ' ')

            continue;

        else if (state == new_set && !strncmp(p, "id=", 3)) {

            void *mem = av_realloc(w->as, sizeof(*w->as) * (w->nb_as + 1));

            if (mem == NULL)

                return AVERROR(ENOMEM);

            w->as = mem;

            ++w->nb_as;

            w->as[w->nb_as - 1].nb_streams = 0;

            w->as[w->nb_as - 1].streams = NULL;

            p += 3; // consume "id="

            q = w->as[w->nb_as - 1].id;

            while (*p != ',') *q++ = *p++;

            *q = 0;

            p++;

            state = parsed_id;

        } else if (state == parsed_id && !strncmp(p, "streams=", 8)) {

            p += 8; // consume "streams="

            state = parsing_streams;

        } else if (state == parsing_streams) {

            struct AdaptationSet *as = &w->as[w->nb_as - 1];

            q = p;

            while (*q != '\0' && *q != ',' && *q != ' ') q++;

            as->streams = av_realloc(as->streams, sizeof(*as->streams) * ++as->nb_streams);

            if (as->streams == NULL)

                return AVERROR(ENOMEM);

            as->streams[as->nb_streams - 1] = to_integer(p, q - p + 1);

            if (as->streams[as->nb_streams - 1] < 0) return -1;

            if (*q == '\0') break;

            if (*q == ' ') state = new_set;

            p = ++q;

        } else {

            return -1;

        }

    }

    return 0;

}
