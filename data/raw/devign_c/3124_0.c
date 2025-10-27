int avformat_open_input(AVFormatContext **ps, const char *filename,

                        AVInputFormat *fmt, AVDictionary **options)

{

    AVFormatContext *s = *ps;

    int ret = 0;

    AVDictionary *tmp = NULL;

    ID3v2ExtraMeta *id3v2_extra_meta = NULL;



    if (!s && !(s = avformat_alloc_context()))

        return AVERROR(ENOMEM);

    if (!s->av_class) {

        av_log(NULL, AV_LOG_ERROR, "Input context has not been properly allocated by avformat_alloc_context() and is not NULL either\n");

        return AVERROR(EINVAL);

    }

    if (fmt)

        s->iformat = fmt;



    if (options)

        av_dict_copy(&tmp, *options, 0);



    if ((ret = av_opt_set_dict(s, &tmp)) < 0)

        goto fail;



    if ((ret = init_input(s, filename, &tmp)) < 0)

        goto fail;

    s->probe_score = ret;

    avio_skip(s->pb, s->skip_initial_bytes);



    /* Check filename in case an image number is expected. */

    if (s->iformat->flags & AVFMT_NEEDNUMBER) {

        if (!av_filename_number_test(filename)) {

            ret = AVERROR(EINVAL);

            goto fail;

        }

    }



    s->duration = s->start_time = AV_NOPTS_VALUE;

    av_strlcpy(s->filename, filename ? filename : "", sizeof(s->filename));



    /* Allocate private data. */

    if (s->iformat->priv_data_size > 0) {

        if (!(s->priv_data = av_mallocz(s->iformat->priv_data_size))) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        if (s->iformat->priv_class) {

            *(const AVClass **) s->priv_data = s->iformat->priv_class;

            av_opt_set_defaults(s->priv_data);

            if ((ret = av_opt_set_dict(s->priv_data, &tmp)) < 0)

                goto fail;

        }

    }



    /* e.g. AVFMT_NOFILE formats will not have a AVIOContext */

    if (s->pb)

        ff_id3v2_read(s, ID3v2_DEFAULT_MAGIC, &id3v2_extra_meta);



    if (!(s->flags&AVFMT_FLAG_PRIV_OPT) && s->iformat->read_header)

        if ((ret = s->iformat->read_header(s)) < 0)

            goto fail;



    if (id3v2_extra_meta) {

        if (!strcmp(s->iformat->name, "mp3") || !strcmp(s->iformat->name, "aac") ||

            !strcmp(s->iformat->name, "tta")) {

            if ((ret = ff_id3v2_parse_apic(s, &id3v2_extra_meta)) < 0)

                goto fail;

        } else

            av_log(s, AV_LOG_DEBUG, "demuxer does not support additional id3 data, skipping\n");

    }

    ff_id3v2_free_extra_meta(&id3v2_extra_meta);



    if ((ret = avformat_queue_attached_pictures(s)) < 0)

        goto fail;



    if (!(s->flags&AVFMT_FLAG_PRIV_OPT) && s->pb && !s->data_offset)

        s->data_offset = avio_tell(s->pb);



    s->raw_packet_buffer_remaining_size = RAW_PACKET_BUFFER_SIZE;



    if (options) {

        av_dict_free(options);

        *options = tmp;

    }

    *ps = s;

    return 0;



fail:

    ff_id3v2_free_extra_meta(&id3v2_extra_meta);

    av_dict_free(&tmp);

    if (s->pb && !(s->flags & AVFMT_FLAG_CUSTOM_IO))

        avio_close(s->pb);

    avformat_free_context(s);

    *ps = NULL;

    return ret;

}
