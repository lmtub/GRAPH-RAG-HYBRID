static int avi_read_tag(AVFormatContext *s, AVStream *st, uint32_t tag, uint32_t size)

{

    AVIOContext *pb = s->pb;

    char key[5] = {0}, *value;



    size += (size & 1);



    if (size == UINT_MAX)

        return -1;

    value = av_malloc(size+1);

    if (!value)

        return -1;

    avio_read(pb, value, size);

    value[size]=0;



    AV_WL32(key, tag);



    return av_dict_set(st ? &st->metadata : &s->metadata, key, value,

                            AV_DICT_DONT_STRDUP_VAL);

}
