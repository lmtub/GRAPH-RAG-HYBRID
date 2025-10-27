static int mov_read_wave(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];



    if((uint64_t)atom.size > (1<<30))

        return -1;



    if (st->codec->codec_id == CODEC_ID_QDM2) {

        // pass all frma atom to codec, needed at least for QDM2

        av_free(st->codec->extradata);

        st->codec->extradata = av_mallocz(atom.size + FF_INPUT_BUFFER_PADDING_SIZE);

        if (!st->codec->extradata)

            return AVERROR(ENOMEM);

        st->codec->extradata_size = atom.size;

        get_buffer(pb, st->codec->extradata, atom.size);

    } else if (atom.size > 8) { /* to read frma, esds atoms */

        if (mov_read_default(c, pb, atom) < 0)

            return -1;

    } else

        url_fskip(pb, atom.size);

    return 0;

}
