static void read_packets(WriterContext *w, AVFormatContext *fmt_ctx)

{

    int i, ret = 0;

    int64_t cur_ts = fmt_ctx->start_time;



    if (read_intervals_nb == 0) {

        ReadInterval interval = (ReadInterval) { .has_start = 0, .has_end = 0 };

        ret = read_interval_packets(w, fmt_ctx, &interval, &cur_ts);

    } else {

        for (i = 0; i < read_intervals_nb; i++) {

            ret = read_interval_packets(w, fmt_ctx, &read_intervals[i], &cur_ts);

            if (ret < 0)

                break;

        }

    }

}
