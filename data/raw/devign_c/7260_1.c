static int parse_playlist(URLContext *h, const char *url)

{

    HLSContext *s = h->priv_data;

    AVIOContext *in;

    int ret = 0, is_segment = 0, is_variant = 0, bandwidth = 0;

    int64_t duration = 0;

    char line[1024];

    const char *ptr;



    if ((ret = avio_open2(&in, url, AVIO_FLAG_READ,

                          &h->interrupt_callback, NULL)) < 0)

        return ret;



    read_chomp_line(in, line, sizeof(line));

    if (strcmp(line, "#EXTM3U"))

        return AVERROR_INVALIDDATA;



    free_segment_list(s);

    s->finished = 0;

    while (!in->eof_reached) {

        read_chomp_line(in, line, sizeof(line));

        if (av_strstart(line, "#EXT-X-STREAM-INF:", &ptr)) {

            struct variant_info info = {{0}};

            is_variant = 1;

            ff_parse_key_value(ptr, (ff_parse_key_val_cb) handle_variant_args,

                               &info);

            bandwidth = atoi(info.bandwidth);

        } else if (av_strstart(line, "#EXT-X-TARGETDURATION:", &ptr)) {

            s->target_duration = atoi(ptr) * AV_TIME_BASE;

        } else if (av_strstart(line, "#EXT-X-MEDIA-SEQUENCE:", &ptr)) {

            s->start_seq_no = atoi(ptr);

        } else if (av_strstart(line, "#EXT-X-ENDLIST", &ptr)) {

            s->finished = 1;

        } else if (av_strstart(line, "#EXTINF:", &ptr)) {

            is_segment = 1;

            duration = atof(ptr) * AV_TIME_BASE;

        } else if (av_strstart(line, "#", NULL)) {

            continue;

        } else if (line[0]) {

            if (is_segment) {

                struct segment *seg = av_malloc(sizeof(struct segment));

                if (!seg) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

                seg->duration = duration;

                ff_make_absolute_url(seg->url, sizeof(seg->url), url, line);

                dynarray_add(&s->segments, &s->n_segments, seg);

                is_segment = 0;

            } else if (is_variant) {

                struct variant *var = av_malloc(sizeof(struct variant));

                if (!var) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

                var->bandwidth = bandwidth;

                ff_make_absolute_url(var->url, sizeof(var->url), url, line);

                dynarray_add(&s->variants, &s->n_variants, var);

                is_variant = 0;

            }

        }

    }

    s->last_load_time = av_gettime_relative();



fail:

    avio_close(in);

    return ret;

}
