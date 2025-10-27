static void flush_buffer(ByteIOContext *s)

{

    if (s->buf_ptr > s->buffer) {

        if (s->write_packet)

            s->write_packet(s->opaque, s->buffer, s->buf_ptr - s->buffer);

        if(s->checksum_ptr){

            s->checksum= s->update_checksum(s->checksum, s->checksum_ptr, s->buf_ptr - s->checksum_ptr);

            s->checksum_ptr= s->buffer;

        }

        s->pos += s->buf_ptr - s->buffer;

    }

    s->buf_ptr = s->buffer;

}
