int64_t avio_seek(AVIOContext *s, int64_t offset, int whence)

{

    int64_t offset1;

    int64_t pos;

    int force = whence & AVSEEK_FORCE;

    whence &= ~AVSEEK_FORCE;



    if(!s)

        return AVERROR(EINVAL);



    pos = s->pos - (s->write_flag ? 0 : (s->buf_end - s->buffer));



    if (whence != SEEK_CUR && whence != SEEK_SET)

        return AVERROR(EINVAL);



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

    } else if ((!s->seekable ||

               offset1 <= s->buf_end + SHORT_SEEK_THRESHOLD - s->buffer) &&

               !s->write_flag && offset1 >= 0 &&

              (whence != SEEK_END || force)) {

        while(s->pos < offset && !s->eof_reached)

            fill_buffer(s);

        if (s->eof_reached)

            return AVERROR_EOF;

        s->buf_ptr = s->buf_end + offset - s->pos;

    } else {

        int64_t res;



        if (s->write_flag) {

            flush_buffer(s);

            s->must_flush = 1;

        }

        if (!s->seek)

            return AVERROR(EPIPE);

        if ((res = s->seek(s->opaque, offset, SEEK_SET)) < 0)

            return res;

        if (!s->write_flag)

            s->buf_end = s->buffer;

        s->buf_ptr = s->buffer;

        s->pos = offset;

    }

    s->eof_reached = 0;

    return offset;

}
