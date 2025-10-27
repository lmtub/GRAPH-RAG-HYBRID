static int get_metadata(AVFormatContext *s,

                        const char *const tag,

                        const unsigned data_size)

{

    uint8_t *buf = ((data_size + 1) == 0) ? NULL : av_malloc(data_size + 1);



    if (!buf)

        return AVERROR(ENOMEM);



    if (avio_read(s->pb, buf, data_size) < 0) {

        av_free(buf);

        return AVERROR(EIO);

    }

    buf[data_size] = 0;

    av_dict_set(&s->metadata, tag, buf, AV_DICT_DONT_STRDUP_VAL);

    return 0;

}
