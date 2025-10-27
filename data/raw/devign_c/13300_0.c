static int mov_read_ctts(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    unsigned int i, j, entries, ctts_count = 0;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];

    sc = st->priv_data;



    avio_r8(pb); /* version */

    avio_rb24(pb); /* flags */

    entries = avio_rb32(pb);



    av_log(c->fc, AV_LOG_TRACE, "track[%u].ctts.entries = %u\n", c->fc->nb_streams - 1, entries);



    if (!entries)

        return 0;

    if (entries >= UINT_MAX / sizeof(*sc->ctts_data))

        return AVERROR_INVALIDDATA;

    av_freep(&sc->ctts_data);

    sc->ctts_data = av_fast_realloc(NULL, &sc->ctts_allocated_size, entries * sizeof(*sc->ctts_data));

    if (!sc->ctts_data)

        return AVERROR(ENOMEM);



    for (i = 0; i < entries && !pb->eof_reached; i++) {

        int count    =avio_rb32(pb);

        int duration =avio_rb32(pb);



        if (count <= 0) {

            av_log(c->fc, AV_LOG_TRACE,

                   "ignoring CTTS entry with count=%d duration=%d\n",

                   count, duration);

            continue;

        }



        /* Expand entries such that we have a 1-1 mapping with samples. */

        for (j = 0; j < count; j++)

            add_ctts_entry(&sc->ctts_data, &ctts_count, &sc->ctts_allocated_size, 1, duration);



        av_log(c->fc, AV_LOG_TRACE, "count=%d, duration=%d\n",

                count, duration);



        if (FFNABS(duration) < -(1<<28) && i+2<entries) {

            av_log(c->fc, AV_LOG_WARNING, "CTTS invalid\n");

            av_freep(&sc->ctts_data);

            sc->ctts_count = 0;

            return 0;

        }



        if (i+2<entries)

            mov_update_dts_shift(sc, duration);

    }



    sc->ctts_count = ctts_count;



    if (pb->eof_reached)

        return AVERROR_EOF;



    av_log(c->fc, AV_LOG_TRACE, "dts shift %d\n", sc->dts_shift);



    return 0;

}
