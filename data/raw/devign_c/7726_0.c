int ff_thread_can_start_frame(AVCodecContext *avctx)

{

    PerThreadContext *p = avctx->thread_opaque;

    if ((avctx->active_thread_type&FF_THREAD_FRAME) && p->state != STATE_SETTING_UP &&

        (avctx->codec->update_thread_context || (!avctx->thread_safe_callbacks &&

                avctx->get_buffer != avcodec_default_get_buffer))) {

        return 0;

    }

    return 1;

}
