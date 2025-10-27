int avformat_open_input(AVFormatContext **ps, const char *filename, AVInputFormat *fmt, AVDictionary **options)

{

    AVFormatContext *s = *ps;

    int ret = 0;

    AVFormatParameters ap = { 0 };

    AVDictionary *tmp = NULL;



    if (!s && !(s = avformat_alloc_context()))

        return AVERROR(ENOMEM);

    if (fmt)

        s->iformat = fmt;



    if (options)

        av_dict_copy(&tmp, *options, 0);



    if ((ret = av_opt_set_dict(s, &tmp)) < 0)

        goto fail;



    if ((ret = init_input(s, filename)) < 0)

        goto fail;



    /* check filename in case an image number is expected */

    if (s->iformat->flags & AVFMT_NEEDNUMBER) {

        if (!av_filename_number_test(filename)) {

            ret = AVERROR(EINVAL);

            goto fail;

        }

    }



    s->duration = s->start_time = AV_NOPTS_VALUE;

    av_strlcpy(s->filename, filename, sizeof(s->filename));



    /* allocate private data */

    if (s->iformat->priv_data_size > 0) {

        if (!(s->priv_data = av_mallocz(s->iformat->priv_data_size))) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        if (s->iformat->priv_class) {

            *(const AVClass**)s->priv_data = s->iformat->priv_class;

            av_opt_set_defaults(s->priv_data);

            if ((ret = av_opt_set_dict(s->priv_data, &tmp)) < 0)

                goto fail;

        }

    }



    /* e.g. AVFMT_NOFILE formats will not have a AVIOContext */

    if (s->pb)

        ff_id3v2_read(s, ID3v2_DEFAULT_MAGIC);



    if (s->iformat->read_header)

        if ((ret = s->iformat->read_header(s, &ap)) < 0)

            goto fail;



    if (s->pb && !s->data_offset)

        s->data_offset = avio_tell(s->pb);



    s->raw_packet_buffer_remaining_size = RAW_PACKET_BUFFER_SIZE;



    if (options) {

        av_dict_free(options);

        *options = tmp;

    }

    *ps = s;

    return 0;



fail:

    av_dict_free(&tmp);

    if (s->pb && !(s->flags & AVFMT_FLAG_CUSTOM_IO))

        avio_close(s->pb);

    avformat_free_context(s);

    *ps = NULL;

    return ret;

}
