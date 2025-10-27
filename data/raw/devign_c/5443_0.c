static int mov_read_stps(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    unsigned i, entries;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];

    sc = st->priv_data;



    avio_rb32(pb); // version + flags



    entries = avio_rb32(pb);

    if (entries >= UINT_MAX / sizeof(*sc->stps_data))

        return AVERROR_INVALIDDATA;

    sc->stps_data = av_malloc(entries * sizeof(*sc->stps_data));

    if (!sc->stps_data)

        return AVERROR(ENOMEM);

    sc->stps_count = entries;



    for (i = 0; i < entries; i++) {

        sc->stps_data[i] = avio_rb32(pb);

        //av_dlog(c->fc, "stps %d\n", sc->stps_data[i]);

    }



    return 0;

}
