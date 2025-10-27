static void fill_buffer(AVIOContext *s)

{

    int max_buffer_size = s->max_packet_size ?

                          s->max_packet_size : IO_BUFFER_SIZE;

    uint8_t *dst        = s->buf_end - s->buffer + max_buffer_size < s->buffer_size ?

                          s->buf_end : s->buffer;

    int len             = s->buffer_size - (dst - s->buffer);



    /* can't fill the buffer without read_packet, just set EOF if appropriate */

    if (!s->read_packet && s->buf_ptr >= s->buf_end)

        s->eof_reached = 1;



    /* no need to do anything if EOF already reached */

    if (s->eof_reached)

        return;



    if (s->update_checksum && dst == s->buffer) {

        if (s->buf_end > s->checksum_ptr)

            s->checksum = s->update_checksum(s->checksum, s->checksum_ptr,

                                             s->buf_end - s->checksum_ptr);

        s->checksum_ptr = s->buffer;

    }



    /* make buffer smaller in case it ended up large after probing */

    if (s->read_packet && s->orig_buffer_size && s->buffer_size > s->orig_buffer_size) {

        if (dst == s->buffer && s->buf_ptr != dst) {

            int ret = ffio_set_buf_size(s, s->orig_buffer_size);

            if (ret < 0)

                av_log(s, AV_LOG_WARNING, "Failed to decrease buffer size\n");



            s->checksum_ptr = dst = s->buffer;

        }

        av_assert0(len >= s->orig_buffer_size);

        len = s->orig_buffer_size;

    }



    if (s->read_packet)

        len = s->read_packet(s->opaque, dst, len);

    else

        len = AVERROR_EOF;

    if (len == AVERROR_EOF) {

        /* do not modify buffer if EOF reached so that a seek back can

           be done without rereading data */

        s->eof_reached = 1;

    } else if (len < 0) {

        s->eof_reached = 1;

        s->error= len;

    } else {

        s->pos += len;

        s->buf_ptr = dst;

        s->buf_end = dst + len;

        s->bytes_read += len;

    }

}
