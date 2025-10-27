static void vp8_filter_mb_row(AVCodecContext *avctx, void *tdata,

                              int jobnr, int threadnr)

{

    VP8Context *s = avctx->priv_data;

    VP8ThreadData *td = &s->thread_data[threadnr];

    int mb_x, mb_y = td->thread_mb_pos >> 16, num_jobs = s->num_jobs;

    AVFrame *curframe = s->curframe->tf.f;

    VP8Macroblock *mb;

    VP8ThreadData *prev_td, *next_td;

    uint8_t *dst[3] = {

        curframe->data[0] + 16 * mb_y * s->linesize,

        curframe->data[1] +  8 * mb_y * s->uvlinesize,

        curframe->data[2] +  8 * mb_y * s->uvlinesize

    };



    if (s->mb_layout == 1)

        mb = s->macroblocks_base + ((s->mb_width + 1) * (mb_y + 1) + 1);

    else

        mb = s->macroblocks + (s->mb_height - mb_y - 1) * 2;



    if (mb_y == 0)

        prev_td = td;

    else

        prev_td = &s->thread_data[(jobnr + num_jobs - 1) % num_jobs];

    if (mb_y == s->mb_height - 1)

        next_td = td;

    else

        next_td = &s->thread_data[(jobnr + 1) % num_jobs];



    for (mb_x = 0; mb_x < s->mb_width; mb_x++, mb++) {

        VP8FilterStrength *f = &td->filter_strength[mb_x];

        if (prev_td != td)

            check_thread_pos(td, prev_td,

                             (mb_x + 1) + (s->mb_width + 3), mb_y - 1);

        if (next_td != td)

            if (next_td != &s->thread_data[0])

                check_thread_pos(td, next_td, mb_x + 1, mb_y + 1);



        if (num_jobs == 1) {

            if (s->filter.simple)

                backup_mb_border(s->top_border[mb_x + 1], dst[0],

                                 NULL, NULL, s->linesize, 0, 1);

            else

                backup_mb_border(s->top_border[mb_x + 1], dst[0],

                                 dst[1], dst[2], s->linesize, s->uvlinesize, 0);

        }



        if (s->filter.simple)

            filter_mb_simple(s, dst[0], f, mb_x, mb_y);

        else

            filter_mb(s, dst, f, mb_x, mb_y);

        dst[0] += 16;

        dst[1] += 8;

        dst[2] += 8;



        update_pos(td, mb_y, (s->mb_width + 3) + mb_x);

    }

}
