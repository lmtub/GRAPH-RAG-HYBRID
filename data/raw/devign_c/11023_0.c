static int copy_to_pbr(DCAXllDecoder *s, uint8_t *data, int size, int delay)

{

    if (size > DCA_XLL_PBR_BUFFER_MAX)

        return AVERROR(ENOSPC);



    if (!s->pbr_buffer && !(s->pbr_buffer = av_malloc(DCA_XLL_PBR_BUFFER_MAX + DCA_BUFFER_PADDING_SIZE)))

        return AVERROR(ENOMEM);



    memcpy(s->pbr_buffer, data, size);

    s->pbr_length = size;

    s->pbr_delay = delay;

    return 0;

}
