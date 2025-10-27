static int mov_read_stsc(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = st->priv_data;

    unsigned int i, entries;



    get_byte(pb); /* version */

    get_be24(pb); /* flags */



    entries = get_be32(pb);



    dprintf(c->fc, "track[%i].stsc.entries = %i\n", c->fc->nb_streams-1, entries);



    if(entries >= UINT_MAX / sizeof(*sc->stsc_data))

        return -1;

    sc->stsc_data = av_malloc(entries * sizeof(*sc->stsc_data));

    if (!sc->stsc_data)

        return AVERROR(ENOMEM);

    sc->stsc_count = entries;



    for(i=0; i<entries; i++) {

        sc->stsc_data[i].first = get_be32(pb);

        sc->stsc_data[i].count = get_be32(pb);

        sc->stsc_data[i].id = get_be32(pb);

    }

    return 0;

}
