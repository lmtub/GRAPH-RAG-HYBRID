static int mov_read_mdhd(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    int version;

    char language[4] = {0};

    unsigned lang;

    int64_t creation_time;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];

    sc = st->priv_data;



    if (sc->time_scale) {

        av_log(c->fc, AV_LOG_ERROR, "Multiple mdhd?\n");

        return AVERROR_INVALIDDATA;

    }



    version = avio_r8(pb);

    if (version > 1) {

        avpriv_request_sample(c->fc, "Version %d", version);

        return AVERROR_PATCHWELCOME;

    }

    avio_rb24(pb); /* flags */

    if (version == 1) {

        creation_time = avio_rb64(pb);

        avio_rb64(pb);

    } else {

        creation_time = avio_rb32(pb);

        avio_rb32(pb); /* modification time */

    }

    mov_metadata_creation_time(&st->metadata, creation_time);



    sc->time_scale = avio_rb32(pb);

    if (sc->time_scale <= 0) {

        av_log(c->fc, AV_LOG_ERROR, "Invalid mdhd time scale %d\n", sc->time_scale);

        return AVERROR_INVALIDDATA;

    }

    st->duration = (version == 1) ? avio_rb64(pb) : avio_rb32(pb); /* duration */



    lang = avio_rb16(pb); /* language */

    if (ff_mov_lang_to_iso639(lang, language))

        av_dict_set(&st->metadata, "language", language, 0);

    avio_rb16(pb); /* quality */



    return 0;

}
