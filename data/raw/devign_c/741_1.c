int64_t avio_seek(AVIOContext *s, int64_t offset, int whence)

{

    int64_t offset1;

    int64_t pos;

    int force = whence & AVSEEK_FORCE;

    int buffer_size;

    int short_seek;

    whence &= ~AVSEEK_FORCE;



    if(!s)




    buffer_size = s->buf_end - s->buffer;

    // pos is the absolute position that the beginning of s->buffer corresponds to in the file

    pos = s->pos - (s->write_flag ? 0 : buffer_size);



    if (whence != SEEK_CUR && whence != SEEK_SET)




    if (whence == SEEK_CUR) {

        offset1 = pos + (s->buf_ptr - s->buffer);

        if (offset == 0)

            return offset1;



        offset += offset1;

    }

    if (offset < 0)




    if (s->short_seek_get) {

        short_seek = s->short_seek_get(s->opaque);

        /* fallback to default short seek */

        if (short_seek <= 0)

            short_seek = s->short_seek_threshold;

    } else

        short_seek = s->short_seek_threshold;



    offset1 = offset - pos; // "offset1" is the relative offset from the beginning of s->buffer

    s->buf_ptr_max = FFMAX(s->buf_ptr_max, s->buf_ptr);

    if ((!s->direct || !s->seek) &&

        offset1 >= 0 && offset1 <= (s->write_flag ? s->buf_ptr_max - s->buffer : buffer_size)) {

        /* can do the seek inside the buffer */

        s->buf_ptr = s->buffer + offset1;

    } else if ((!(s->seekable & AVIO_SEEKABLE_NORMAL) ||

               offset1 <= buffer_size + short_seek) &&

               !s->write_flag && offset1 >= 0 &&

               (!s->direct || !s->seek) &&

              (whence != SEEK_END || force)) {

        while(s->pos < offset && !s->eof_reached)

            fill_buffer(s);

        if (s->eof_reached)

            return AVERROR_EOF;

        s->buf_ptr = s->buf_end - (s->pos - offset);

    } else if(!s->write_flag && offset1 < 0 && -offset1 < buffer_size>>1 && s->seek && offset > 0) {

        int64_t res;



        pos -= FFMIN(buffer_size>>1, pos);

        if ((res = s->seek(s->opaque, pos, SEEK_SET)) < 0)

            return res;

        s->buf_end =

        s->buf_ptr = s->buffer;

        s->pos = pos;

        s->eof_reached = 0;

        fill_buffer(s);

        return avio_seek(s, offset, SEEK_SET | force);

    } else {

        int64_t res;

        if (s->write_flag) {

            flush_buffer(s);

        }

        if (!s->seek)

            return AVERROR(EPIPE);

        if ((res = s->seek(s->opaque, offset, SEEK_SET)) < 0)

            return res;

        s->seek_count ++;

        if (!s->write_flag)

            s->buf_end = s->buffer;

        s->buf_ptr = s->buf_ptr_max = s->buffer;

        s->pos = offset;

    }

    s->eof_reached = 0;

    return offset;

}