static int buf_put_buffer(void *opaque, const uint8_t *buf,

                          int64_t pos, int size)

{

    QEMUBuffer *s = opaque;



    return qsb_write_at(s->qsb, buf, pos, size);

}
