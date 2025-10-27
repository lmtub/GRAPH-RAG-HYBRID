static int mov_read_glbl(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];



    if ((uint64_t)atom.size > (1<<30))

        return AVERROR_INVALIDDATA;



    if (atom.size >= 10) {

        // Broken files created by legacy versions of libavformat will

        // wrap a whole fiel atom inside of a glbl atom.

        unsigned size = avio_rb32(pb);

        unsigned type = avio_rl32(pb);

        avio_seek(pb, -8, SEEK_CUR);

        if (type == MKTAG('f','i','e','l') && size == atom.size)

            return mov_read_default(c, pb, atom);

    }

    av_free(st->codec->extradata);

    st->codec->extradata = av_mallocz(atom.size + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!st->codec->extradata)

        return AVERROR(ENOMEM);

    st->codec->extradata_size = atom.size;

    avio_read(pb, st->codec->extradata, atom.size);

    return 0;

}
