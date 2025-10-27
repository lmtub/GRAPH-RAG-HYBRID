static int mov_read_stss(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = st->priv_data;

    unsigned int i, entries;



    get_byte(pb); /* version */

    get_be24(pb); /* flags */



    entries = get_be32(pb);



    dprintf(c->fc, "keyframe_count = %d\n", entries);



    if(entries >= UINT_MAX / sizeof(int))

        return -1;

    sc->keyframes = av_malloc(entries * sizeof(int));

    if (!sc->keyframes)

        return AVERROR(ENOMEM);

    sc->keyframe_count = entries;



    for(i=0; i<entries; i++) {

        sc->keyframes[i] = get_be32(pb);

        //dprintf(c->fc, "keyframes[]=%d\n", sc->keyframes[i]);

    }

    return 0;

}
