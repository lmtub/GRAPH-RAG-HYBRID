void ff_thread_flush(AVCodecContext *avctx)

{

    FrameThreadContext *fctx = avctx->thread_opaque;



    if (!avctx->thread_opaque) return;



    park_frame_worker_threads(fctx, avctx->thread_count);



    if (fctx->prev_thread)

        update_context_from_thread(fctx->threads->avctx, fctx->prev_thread->avctx, 0);



    fctx->next_decoding = fctx->next_finished = 0;

    fctx->delaying = 1;

    fctx->prev_thread = NULL;

}
