static void rm_read_metadata(AVFormatContext *s, int wide)

{

    char buf[1024];

    int i;

    for (i=0; i<FF_ARRAY_ELEMS(ff_rm_metadata); i++) {

        int len = wide ? avio_rb16(s->pb) : avio_r8(s->pb);

        get_strl(s->pb, buf, sizeof(buf), len);

        av_dict_set(&s->metadata, ff_rm_metadata[i], buf, 0);

    }

}
