static int mov_read_ctts(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = st->priv_data;

    unsigned int i, entries;



    get_byte(pb); /* version */

    get_be24(pb); /* flags */

    entries = get_be32(pb);

    if(entries >= UINT_MAX / sizeof(MOV_stts_t))

        return -1;



    sc->ctts_count = entries;

    sc->ctts_data = av_malloc(entries * sizeof(MOV_stts_t));

    if (!sc->ctts_data)

        return -1;

    dprintf(c->fc, "track[%i].ctts.entries = %i\n", c->fc->nb_streams-1, entries);



    for(i=0; i<entries; i++) {

        int count    =get_be32(pb);

        int duration =get_be32(pb);



        if (duration < 0) {

            av_log(c->fc, AV_LOG_ERROR, "negative ctts, ignoring\n");

            sc->ctts_count = 0;

            url_fskip(pb, 8 * (entries - i - 1));

            break;

        }

        sc->ctts_data[i].count   = count;

        sc->ctts_data[i].duration= duration;



        sc->time_rate= ff_gcd(sc->time_rate, duration);

    }

    return 0;

}
