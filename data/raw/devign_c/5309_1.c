static int parse_playlist(HLSContext *c, const char *url,

                          struct variant *var, AVIOContext *in)

{

    int ret = 0, is_segment = 0, is_variant = 0, bandwidth = 0;

    int64_t duration = 0;

    enum KeyType key_type = KEY_NONE;

    uint8_t iv[16] = "";

    int has_iv = 0;

    char key[MAX_URL_SIZE] = "";

    char line[1024];

    const char *ptr;

    int close_in = 0;



    if (!in) {

        AVDictionary *opts = NULL;

        close_in = 1;

        /* Some HLS servers don't like being sent the range header */

        av_dict_set(&opts, "seekable", "0", 0);



        // broker prior HTTP options that should be consistent across requests

        av_dict_set(&opts, "user-agent", c->user_agent, 0);

        av_dict_set(&opts, "cookies", c->cookies, 0);



        ret = avio_open2(&in, url, AVIO_FLAG_READ,

                         c->interrupt_callback, &opts);

        av_dict_free(&opts);

        if (ret < 0)

            return ret;

    }



    read_chomp_line(in, line, sizeof(line));

    if (strcmp(line, "#EXTM3U")) {

        ret = AVERROR_INVALIDDATA;

        goto fail;

    }



    if (var) {

        free_segment_list(var);

        var->finished = 0;

    }

    while (!url_feof(in)) {

        read_chomp_line(in, line, sizeof(line));

        if (av_strstart(line, "#EXT-X-STREAM-INF:", &ptr)) {

            struct variant_info info = {{0}};

            is_variant = 1;

            ff_parse_key_value(ptr, (ff_parse_key_val_cb) handle_variant_args,

                               &info);

            bandwidth = atoi(info.bandwidth);

        } else if (av_strstart(line, "#EXT-X-KEY:", &ptr)) {

            struct key_info info = {{0}};

            ff_parse_key_value(ptr, (ff_parse_key_val_cb) handle_key_args,

                               &info);

            key_type = KEY_NONE;

            has_iv = 0;

            if (!strcmp(info.method, "AES-128"))

                key_type = KEY_AES_128;

            if (!strncmp(info.iv, "0x", 2) || !strncmp(info.iv, "0X", 2)) {

                ff_hex_to_data(iv, info.iv + 2);

                has_iv = 1;

            }

            av_strlcpy(key, info.uri, sizeof(key));

        } else if (av_strstart(line, "#EXT-X-TARGETDURATION:", &ptr)) {

            if (!var) {

                var = new_variant(c, 0, url, NULL);

                if (!var) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

            }

            var->target_duration = atoi(ptr) * AV_TIME_BASE;

        } else if (av_strstart(line, "#EXT-X-MEDIA-SEQUENCE:", &ptr)) {

            if (!var) {

                var = new_variant(c, 0, url, NULL);

                if (!var) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

            }

            var->start_seq_no = atoi(ptr);

        } else if (av_strstart(line, "#EXT-X-ENDLIST", &ptr)) {

            if (var)

                var->finished = 1;

        } else if (av_strstart(line, "#EXTINF:", &ptr)) {

            is_segment = 1;

            duration   = atof(ptr) * AV_TIME_BASE;

        } else if (av_strstart(line, "#", NULL)) {

            continue;

        } else if (line[0]) {

            if (is_variant) {

                if (!new_variant(c, bandwidth, line, url)) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

                is_variant = 0;

                bandwidth  = 0;

            }

            if (is_segment) {

                struct segment *seg;

                if (!var) {

                    var = new_variant(c, 0, url, NULL);

                    if (!var) {

                        ret = AVERROR(ENOMEM);

                        goto fail;

                    }

                }

                seg = av_malloc(sizeof(struct segment));

                if (!seg) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

                seg->duration = duration;

                seg->key_type = key_type;

                if (has_iv) {

                    memcpy(seg->iv, iv, sizeof(iv));

                } else {

                    int seq = var->start_seq_no + var->n_segments;

                    memset(seg->iv, 0, sizeof(seg->iv));

                    AV_WB32(seg->iv + 12, seq);

                }

                ff_make_absolute_url(seg->key, sizeof(seg->key), url, key);

                ff_make_absolute_url(seg->url, sizeof(seg->url), url, line);

                dynarray_add(&var->segments, &var->n_segments, seg);

                is_segment = 0;

            }

        }

    }

    if (var)

        var->last_load_time = av_gettime();



fail:

    if (close_in)

        avio_close(in);

    return ret;

}
