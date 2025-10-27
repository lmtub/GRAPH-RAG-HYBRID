void ff_thread_flush(AVCodecContext *avctx)

{

    FrameThreadContext *fctx = avctx->thread_opaque;



    if (!avctx->thread_opaque) return;



    park_frame_worker_threads(fctx, avctx->thread_count);

    if (fctx->prev_thread) {

        if (fctx->prev_thread != &fctx->threads[0])

            update_context_from_thread(fctx->threads[0].avctx, fctx->prev_thread->avctx, 0);

        if (avctx->codec->flush)

            avctx->codec->flush(fctx->threads[0].avctx);

    }



    fctx->next_decoding = fctx->next_finished = 0;

    fctx->delaying = 1;

    fctx->prev_thread = NULL;

    // Make sure decode flush calls with size=0 won't return old frames

    for (int i = 0; i < avctx->thread_count; i++)

        fctx->threads[i].got_frame = 0;

}
