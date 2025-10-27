int vp78_decode_mb_row_sliced(AVCodecContext *avctx, void *tdata, int jobnr,

                              int threadnr, int is_vp7)

{

    VP8Context *s = avctx->priv_data;

    VP8ThreadData *td = &s->thread_data[jobnr];

    VP8ThreadData *next_td = NULL, *prev_td = NULL;

    VP8Frame *curframe = s->curframe;

    int mb_y, num_jobs = s->num_jobs;

    int ret;



    td->thread_nr = threadnr;

    for (mb_y = jobnr; mb_y < s->mb_height; mb_y += num_jobs) {

        if (mb_y >= s->mb_height)

            break;

        td->thread_mb_pos = mb_y << 16;

        ret = s->decode_mb_row_no_filter(avctx, tdata, jobnr, threadnr);

        if (ret < 0)

            return ret;

        if (s->deblock_filter)

            s->filter_mb_row(avctx, tdata, jobnr, threadnr);

        update_pos(td, mb_y, INT_MAX & 0xFFFF);



        s->mv_min.y -= 64;

        s->mv_max.y -= 64;



        if (avctx->active_thread_type == FF_THREAD_FRAME)

            ff_thread_report_progress(&curframe->tf, mb_y, 0);

    }



    return 0;

}
