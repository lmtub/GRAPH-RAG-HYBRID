static int h264_init_context(AVCodecContext *avctx, H264Context *h)
{
    int i;
    h->avctx                 = avctx;
    h->picture_structure     = PICT_FRAME;
    h->workaround_bugs       = avctx->workaround_bugs;
    h->flags                 = avctx->flags;
    h->poc.prev_poc_msb      = 1 << 16;
    h->recovery_frame        = -1;
    h->frame_recovered       = 0;
    h->next_outputed_poc = INT_MIN;
    for (i = 0; i < MAX_DELAYED_PIC_COUNT; i++)
        h->last_pocs[i] = INT_MIN;
    ff_h264_sei_uninit(&h->sei);
    avctx->chroma_sample_location = AVCHROMA_LOC_LEFT;
    h->nb_slice_ctx = (avctx->active_thread_type & FF_THREAD_SLICE) ? avctx->thread_count : 1;
    h->slice_ctx = av_mallocz_array(h->nb_slice_ctx, sizeof(*h->slice_ctx));
    if (!h->slice_ctx) {
        h->nb_slice_ctx = 0;
        return AVERROR(ENOMEM);
    }
    for (i = 0; i < H264_MAX_PICTURE_COUNT; i++) {
        h->DPB[i].f = av_frame_alloc();
        if (!h->DPB[i].f)
            return AVERROR(ENOMEM);
    }
    h->cur_pic.f = av_frame_alloc();
    if (!h->cur_pic.f)
        return AVERROR(ENOMEM);
    h->output_frame = av_frame_alloc();
    if (!h->output_frame)
        return AVERROR(ENOMEM);
    for (i = 0; i < h->nb_slice_ctx; i++)
        h->slice_ctx[i].h264 = h;
    return 0;
}