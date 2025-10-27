int ffio_rewind_with_probe_data(AVIOContext *s, unsigned char *buf, int buf_size)

{

    int64_t buffer_start;

    int buffer_size;

    int overlap, new_size, alloc_size;



    if (s->write_flag)

        return AVERROR(EINVAL);



    buffer_size = s->buf_end - s->buffer;



    /* the buffers must touch or overlap */

    if ((buffer_start = s->pos - buffer_size) > buf_size)

        return AVERROR(EINVAL);



    overlap = buf_size - buffer_start;

    new_size = buf_size + buffer_size - overlap;



    alloc_size = FFMAX(s->buffer_size, new_size);

    if (alloc_size > buf_size)

        if (!(buf = av_realloc_f(buf, 1, alloc_size)))

            return AVERROR(ENOMEM);



    if (new_size > buf_size) {

        memcpy(buf + buf_size, s->buffer + overlap, buffer_size - overlap);

        buf_size = new_size;

    }



    av_free(s->buffer);

    s->buf_ptr = s->buffer = buf;

    s->buffer_size = alloc_size;

    s->pos = buf_size;

    s->buf_end = s->buf_ptr + buf_size;

    s->eof_reached = 0;

    s->must_flush = 0;



    return 0;

}
