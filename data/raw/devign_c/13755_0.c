static AVIndexEntry *mov_find_next_sample(AVFormatContext *s, AVStream **st)

{

    AVIndexEntry *sample = NULL;

    int64_t best_dts = INT64_MAX;

    int i;

    for (i = 0; i < s->nb_streams; i++) {

        AVStream *avst = s->streams[i];

        MOVStreamContext *msc = avst->priv_data;

        if (msc->pb && msc->current_sample < avst->nb_index_entries) {

            AVIndexEntry *current_sample = &avst->index_entries[msc->current_sample];

            int64_t dts;

            if (msc->ctts_data)

                dts = av_rescale(current_sample->timestamp - msc->dts_shift - msc->ctts_data[msc->ctts_index].duration,

                                 AV_TIME_BASE, msc->time_scale);

            else

                dts = av_rescale(current_sample->timestamp, AV_TIME_BASE, msc->time_scale);

            av_dlog(s, "stream %d, sample %d, dts %"PRId64"\n", i, msc->current_sample, dts);

            if (!sample || (!s->pb->seekable && current_sample->pos < sample->pos) ||

                (s->pb->seekable &&

                 ((msc->pb != s->pb && dts < best_dts) || (msc->pb == s->pb &&

                 ((FFABS(best_dts - dts) <= AV_TIME_BASE && current_sample->pos < sample->pos) ||

                  (FFABS(best_dts - dts) > AV_TIME_BASE && dts < best_dts)))))) {

                sample = current_sample;

                best_dts = dts;

                *st = avst;

            }

        }

    }

    return sample;

}
