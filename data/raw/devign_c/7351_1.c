offset_t url_fseek(ByteIOContext *s, offset_t offset, int whence)

{

    offset_t offset1;

    offset_t pos= s->pos - (s->write_flag ? 0 : (s->buf_end - s->buffer));



    if (whence != SEEK_CUR && whence != SEEK_SET)

        return -EINVAL;



    if (whence == SEEK_CUR) {

        offset1 = pos + (s->buf_ptr - s->buffer);

        if (offset == 0)

            return offset1;

        offset += offset1;

    }

    offset1 = offset - pos;

    if (!s->must_flush &&

        offset1 >= 0 && offset1 < (s->buf_end - s->buffer)) {

        /* can do the seek inside the buffer */

        s->buf_ptr = s->buffer + offset1;

    } else {

        if (!s->seek)

            return -EPIPE;



#ifdef CONFIG_MUXERS

        if (s->write_flag) {

            flush_buffer(s);

            s->must_flush = 1;

        } else

#endif //CONFIG_MUXERS

        {

            s->buf_end = s->buffer;

        }

        s->buf_ptr = s->buffer;

        if (s->seek(s->opaque, offset, SEEK_SET) == (offset_t)-EPIPE)

            return -EPIPE;

        s->pos = offset;

    }

    s->eof_reached = 0;

    return offset;

}
