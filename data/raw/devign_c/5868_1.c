int ff_mpv_reallocate_putbitbuffer(MpegEncContext *s, size_t threshold, size_t size_increase)
{
    if (   s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb)>>3) < threshold
        && s->slice_context_count == 1
        && s->pb.buf == s->avctx->internal->byte_buffer) {
        int lastgob_pos = s->ptr_lastgob - s->pb.buf;
        int vbv_pos     = s->vbv_delay_ptr - s->pb.buf;
        uint8_t *new_buffer = NULL;
        int new_buffer_size = 0;
        av_fast_padded_malloc(&new_buffer, &new_buffer_size,
                              s->avctx->internal->byte_buffer_size + size_increase);
        if (!new_buffer)
        memcpy(new_buffer, s->avctx->internal->byte_buffer, s->avctx->internal->byte_buffer_size);
        av_free(s->avctx->internal->byte_buffer);
        s->avctx->internal->byte_buffer      = new_buffer;
        s->avctx->internal->byte_buffer_size = new_buffer_size;
        rebase_put_bits(&s->pb, new_buffer, new_buffer_size);
        s->ptr_lastgob   = s->pb.buf + lastgob_pos;
        s->vbv_delay_ptr = s->pb.buf + vbv_pos;
    if (s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb)>>3) < threshold)
        return AVERROR(EINVAL);
    return 0;