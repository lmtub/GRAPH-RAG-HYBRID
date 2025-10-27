int ffio_read_indirect(AVIOContext *s, unsigned char *buf, int size, unsigned char **data)

{

    if (s->buf_end - s->buf_ptr >= size && !s->write_flag) {

        *data = s->buf_ptr;

        s->buf_ptr += size;

        return size;

    } else {

        *data = buf;

        return avio_read(s, buf, size);

    }

}
