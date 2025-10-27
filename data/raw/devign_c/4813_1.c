static int asf_read_generic_value(AVFormatContext *s, uint8_t *name, uint16_t name_len,

                                  int type, AVDictionary **met)

{

    AVIOContext *pb = s->pb;

    uint64_t value;

    char buf[32];



    switch (type) {

    case ASF_BOOL:

        value = avio_rl32(pb);

        break;

    case ASF_DWORD:

        value = avio_rl32(pb);

        break;

    case ASF_QWORD:

        value = avio_rl64(pb);

        break;

    case ASF_WORD:

        value = avio_rl16(pb);

        break;

    default:

        av_freep(&name);

        return AVERROR_INVALIDDATA;

    }

    snprintf(buf, sizeof(buf), "%"PRIu64, value);

    if (av_dict_set(met, name, buf, 0) < 0)

        av_log(s, AV_LOG_WARNING, "av_dict_set failed.\n");



    return 0;

}
