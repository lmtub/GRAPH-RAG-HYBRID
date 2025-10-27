static int mov_read_elst(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    MOVStreamContext *sc;

    int i, edit_count, version;



    if (c->fc->nb_streams < 1 || c->ignore_editlist)

        return 0;

    sc = c->fc->streams[c->fc->nb_streams-1]->priv_data;



    version = avio_r8(pb); /* version */

    avio_rb24(pb); /* flags */

    edit_count = avio_rb32(pb); /* entries */



    if (!edit_count)

        return 0;

    if (sc->elst_data)

        av_log(c->fc, AV_LOG_WARNING, "Duplicated ELST atom\n");

    av_free(sc->elst_data);

    sc->elst_count = 0;

    sc->elst_data = av_malloc_array(edit_count, sizeof(*sc->elst_data));

    if (!sc->elst_data)

        return AVERROR(ENOMEM);



    av_log(c->fc, AV_LOG_TRACE, "track[%u].edit_count = %i\n", c->fc->nb_streams - 1, edit_count);

    for (i = 0; i < edit_count && !pb->eof_reached; i++) {

        MOVElst *e = &sc->elst_data[i];



        if (version == 1) {

            e->duration = avio_rb64(pb);

            e->time     = avio_rb64(pb);

        } else {

            e->duration = avio_rb32(pb); /* segment duration */

            e->time     = (int32_t)avio_rb32(pb); /* media time */

        }

        e->rate = avio_rb32(pb) / 65536.0;

        av_log(c->fc, AV_LOG_TRACE, "duration=%"PRId64" time=%"PRId64" rate=%f\n",

               e->duration, e->time, e->rate);



        if (e->time < 0 && e->time != -1 &&

            c->fc->strict_std_compliance >= FF_COMPLIANCE_STRICT) {

            av_log(c->fc, AV_LOG_ERROR, "Track %d, edit %d: Invalid edit list media time=%"PRId64"\n",

                   c->fc->nb_streams-1, i, e->time);

            return AVERROR_INVALIDDATA;

        }

    }

    sc->elst_count = i;



    return 0;

}
