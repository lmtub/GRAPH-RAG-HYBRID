static void hevc_await_progress(HEVCContext *s, HEVCFrame *ref,

                                const Mv *mv, int y0, int height)

{

    int y = FFMAX(0, (mv->y >> 2) + y0 + height + 9);



    if (s->threads_type == FF_THREAD_FRAME )

        ff_thread_await_progress(&ref->tf, y, 0);

}
