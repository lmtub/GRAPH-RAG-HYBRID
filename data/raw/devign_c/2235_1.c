static int decode_init_thread_copy(AVCodecContext *avctx)

{

    H264Context *h = avctx->priv_data;



    if (!avctx->internal->is_copy)

        return 0;

    memset(h->sps_buffers, 0, sizeof(h->sps_buffers));

    memset(h->pps_buffers, 0, sizeof(h->pps_buffers));




    h->rbsp_buffer[0]      = NULL;

    h->rbsp_buffer[1]      = NULL;

    h->rbsp_buffer_size[0] = 0;

    h->rbsp_buffer_size[1] = 0;

    h->context_initialized = 0;



    return 0;

}