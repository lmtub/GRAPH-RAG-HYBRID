static int open_url(AVFormatContext *s, AVIOContext **pb, const char *url,

                    AVDictionary *opts, AVDictionary *opts2)

{

    HLSContext *c = s->priv_data;

    AVDictionary *tmp = NULL;

    const char *proto_name = avio_find_protocol_name(url);

    int ret;



    av_dict_copy(&tmp, opts, 0);

    av_dict_copy(&tmp, opts2, 0);



    if (!proto_name)

        return AVERROR_INVALIDDATA;



    // only http(s) & file are allowed

    if (!av_strstart(proto_name, "http", NULL) && !av_strstart(proto_name, "file", NULL))

        return AVERROR_INVALIDDATA;

    if (!strncmp(proto_name, url, strlen(proto_name)) && url[strlen(proto_name)] == ':')

        ;

    else if (strcmp(proto_name, "file") || !strncmp(url, "file,", 5))

        return AVERROR_INVALIDDATA;



    ret = s->io_open(s, pb, url, AVIO_FLAG_READ, &tmp);

    if (ret >= 0) {

        // update cookies on http response with setcookies.

        void *u = (s->flags & AVFMT_FLAG_CUSTOM_IO) ? NULL : s->pb;

        update_options(&c->cookies, "cookies", u);

        av_dict_set(&opts, "cookies", c->cookies, 0);

    }



    av_dict_free(&tmp);



    return ret;

}
