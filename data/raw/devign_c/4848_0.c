void ff_hevc_hls_filter(HEVCContext *s, int x, int y, int ctb_size)

{

    deblocking_filter_CTB(s, x, y);

    if (s->sps->sao_enabled) {

        int x_end = x >= s->sps->width  - ctb_size;

        int y_end = y >= s->sps->height - ctb_size;

        if (y && x)

            sao_filter_CTB(s, x - ctb_size, y - ctb_size);

        if (x && y_end)

            sao_filter_CTB(s, x - ctb_size, y);

        if (y && x_end) {

            sao_filter_CTB(s, x, y - ctb_size);

            if (s->threads_type & FF_THREAD_FRAME )

                ff_thread_report_progress(&s->ref->tf, y - ctb_size, 0);

        }

        if (x_end && y_end) {

            sao_filter_CTB(s, x , y);

            if (s->threads_type & FF_THREAD_FRAME )

                ff_thread_report_progress(&s->ref->tf, y, 0);

        }

    } else {

        if (y && x >= s->sps->width - ctb_size)

            if (s->threads_type & FF_THREAD_FRAME )

                ff_thread_report_progress(&s->ref->tf, y, 0);

    }

}
