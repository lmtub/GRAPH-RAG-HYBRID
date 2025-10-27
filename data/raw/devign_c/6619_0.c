int ffio_ensure_seekback(AVIOContext *s, int buf_size)

{

    uint8_t *buffer;

    int max_buffer_size = s->max_packet_size ?

                          s->max_packet_size : IO_BUFFER_SIZE;



    buf_size += s->buf_ptr - s->buffer + max_buffer_size;



    if (buf_size < s->buffer_size || s->seekable)

        return 0;

    av_assert0(!s->write_flag);



    buffer = av_malloc(buf_size);

    if (!buffer)

        return AVERROR(ENOMEM);



    memcpy(buffer, s->buffer, s->buffer_size);

    av_free(s->buffer);

    s->buf_ptr = buffer + (s->buf_ptr - s->buffer);

    s->buf_end = buffer + (s->buf_end - s->buffer);

    s->buffer = buffer;

    s->buffer_size = buf_size;

    return 0;

}
