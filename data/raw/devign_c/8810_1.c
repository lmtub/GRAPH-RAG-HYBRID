static int mov_read_extradata(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    uint64_t size;

    uint8_t *buf;

    int err;



    if (c->fc->nb_streams < 1) // will happen with jp2 files

        return 0;

    st= c->fc->streams[c->fc->nb_streams-1];

    size= (uint64_t)st->codec->extradata_size + atom.size + 8 + FF_INPUT_BUFFER_PADDING_SIZE;

    if (size > INT_MAX || (uint64_t)atom.size > INT_MAX)

        return AVERROR_INVALIDDATA;

    if ((err = av_reallocp(&st->codec->extradata, size)) < 0) {

        st->codec->extradata_size = 0;

        return err;

    }

    buf = st->codec->extradata + st->codec->extradata_size;

    st->codec->extradata_size= size - FF_INPUT_BUFFER_PADDING_SIZE;

    AV_WB32(       buf    , atom.size + 8);

    AV_WL32(       buf + 4, atom.type);

    avio_read(pb, buf + 8, atom.size);

    return 0;

}
