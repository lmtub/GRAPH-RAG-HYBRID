static int mov_seek_stream(AVStream *st, int64_t timestamp, int flags)

{

    MOVStreamContext *sc = st->priv_data;

    int sample, time_sample;

    int i;



    sample = av_index_search_timestamp(st, timestamp, flags);

    dprintf(st->codec, "stream %d, timestamp %"PRId64", sample %d\n", st->index, timestamp, sample);

    if (sample < 0) /* not sure what to do */

        return -1;

    sc->current_sample = sample;

    dprintf(st->codec, "stream %d, found sample %d\n", st->index, sc->current_sample);

    /* adjust ctts index */

    if (sc->ctts_data) {

        time_sample = 0;

        for (i = 0; i < sc->ctts_count; i++) {

            time_sample += sc->ctts_data[i].count;

            if (time_sample >= sc->current_sample) {

                sc->sample_to_ctime_index = i;

                sc->sample_to_ctime_sample = time_sample - sc->current_sample;

                break;

            }

        }

    }

    return sample;

}
