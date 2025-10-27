static int mov_read_hdlr(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    uint32_t type;

    uint32_t av_unused ctype;

    int64_t title_size;

    char *title_str;



    if (c->fc->nb_streams < 1) // meta before first trak

        return 0;



    st = c->fc->streams[c->fc->nb_streams-1];



    avio_r8(pb); /* version */

    avio_rb24(pb); /* flags */



    /* component type */

    ctype = avio_rl32(pb);

    type = avio_rl32(pb); /* component subtype */



    av_log(c->fc, AV_LOG_TRACE, "ctype= %.4s (0x%08x)\n", (char*)&ctype, ctype);

    av_log(c->fc, AV_LOG_TRACE, "stype= %.4s\n", (char*)&type);



    if     (type == MKTAG('v','i','d','e'))

        st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    else if (type == MKTAG('s','o','u','n'))

        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

    else if (type == MKTAG('m','1','a',' '))

        st->codec->codec_id = AV_CODEC_ID_MP2;

    else if ((type == MKTAG('s','u','b','p')) || (type == MKTAG('c','l','c','p')))

        st->codec->codec_type = AVMEDIA_TYPE_SUBTITLE;



    avio_rb32(pb); /* component  manufacture */

    avio_rb32(pb); /* component flags */

    avio_rb32(pb); /* component flags mask */



    title_size = atom.size - 24;

    if (title_size > 0) {

        title_str = av_malloc(title_size + 1); /* Add null terminator */

        if (!title_str)

            return AVERROR(ENOMEM);

        avio_read(pb, title_str, title_size);

        title_str[title_size] = 0;

        if (title_str[0]) {

            int off = (!c->isom && title_str[0] == title_size - 1);

            av_dict_set(&st->metadata, "handler_name", title_str + off, 0);

        }

        av_freep(&title_str);

    }



    return 0;

}
