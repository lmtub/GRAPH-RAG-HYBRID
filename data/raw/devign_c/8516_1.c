static int applehttp_open(URLContext *h, const char *uri, int flags)

{

    AppleHTTPContext *s;

    int ret, i;

    const char *nested_url;



    if (flags & AVIO_FLAG_WRITE)

        return AVERROR(ENOSYS);



    s = av_mallocz(sizeof(AppleHTTPContext));

    if (!s)

        return AVERROR(ENOMEM);

    h->priv_data = s;

    h->is_streamed = 1;



    if (av_strstart(uri, "applehttp+", &nested_url)) {

        av_strlcpy(s->playlisturl, nested_url, sizeof(s->playlisturl));

    } else if (av_strstart(uri, "applehttp://", &nested_url)) {

        av_strlcpy(s->playlisturl, "http://", sizeof(s->playlisturl));

        av_strlcat(s->playlisturl, nested_url, sizeof(s->playlisturl));

    } else {

        av_log(h, AV_LOG_ERROR, "Unsupported url %s\n", uri);

        ret = AVERROR(EINVAL);

        goto fail;

    }



    if ((ret = parse_playlist(h, s->playlisturl)) < 0)

        goto fail;



    if (s->n_segments == 0 && s->n_variants > 0) {

        int max_bandwidth = 0, maxvar = -1;

        for (i = 0; i < s->n_variants; i++) {

            if (s->variants[i]->bandwidth > max_bandwidth || i == 0) {

                max_bandwidth = s->variants[i]->bandwidth;

                maxvar = i;

            }

        }

        av_strlcpy(s->playlisturl, s->variants[maxvar]->url,

                   sizeof(s->playlisturl));

        if ((ret = parse_playlist(h, s->playlisturl)) < 0)

            goto fail;

    }



    if (s->n_segments == 0) {

        av_log(h, AV_LOG_WARNING, "Empty playlist\n");

        ret = AVERROR(EIO);

        goto fail;

    }

    s->cur_seq_no = s->start_seq_no;

    if (!s->finished && s->n_segments >= 3)

        s->cur_seq_no = s->start_seq_no + s->n_segments - 3;



    return 0;



fail:

    av_free(s);

    return ret;

}
