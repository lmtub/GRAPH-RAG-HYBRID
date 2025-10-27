static int mov_read_stsz(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = (MOVStreamContext *)st->priv_data;

    int entries, i;



    print_atom("stsz", atom);



    get_byte(pb); /* version */

    get_byte(pb); get_byte(pb); get_byte(pb); /* flags */



    sc->sample_size = get_be32(pb);

    entries = get_be32(pb);

    sc->sample_count = entries;

#ifdef DEBUG

    av_log(NULL, AV_LOG_DEBUG, "sample_size = %ld sample_count = %ld\n", sc->sample_size, sc->sample_count);

#endif

    if(sc->sample_size)

        return 0; /* there isn't any table following */

    sc->sample_sizes = (long*) av_malloc(entries * sizeof(long));

    if (!sc->sample_sizes)

        return -1;

    for(i=0; i<entries; i++) {

        sc->sample_sizes[i] = get_be32(pb);

#ifdef DEBUG

/*        av_log(NULL, AV_LOG_DEBUG, "sample_sizes[]=%ld\n", sc->sample_sizes[i]); */

#endif

    }

    return 0;

}
