static int mov_read_mdhd(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = st->priv_data;

    int version = get_byte(pb);

    char language[4] = {0};

    unsigned lang;



    if (version > 1)

        return -1; /* unsupported */



    get_be24(pb); /* flags */

    if (version == 1) {

        get_be64(pb);

        get_be64(pb);

    } else {

        get_be32(pb); /* creation time */

        get_be32(pb); /* modification time */

    }



    sc->time_scale = get_be32(pb);

    st->duration = (version == 1) ? get_be64(pb) : get_be32(pb); /* duration */



    lang = get_be16(pb); /* language */

    if (ff_mov_lang_to_iso639(lang, language))

        av_metadata_set(&st->metadata, "language", language);

    get_be16(pb); /* quality */



    return 0;

}
