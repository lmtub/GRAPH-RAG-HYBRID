static int buf_get_buffer(void *opaque, uint8_t *buf, int64_t pos, int size)

{

    QEMUBuffer *s = opaque;

    ssize_t len = qsb_get_length(s->qsb) - pos;



    if (len <= 0) {

        return 0;

    }



    if (len > size) {

        len = size;

    }

    return qsb_get_buffer(s->qsb, pos, len, buf);

}
