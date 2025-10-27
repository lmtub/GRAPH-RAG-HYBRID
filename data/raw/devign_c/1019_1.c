int init_put_byte(ByteIOContext *s,

                  unsigned char *buffer,

                  int buffer_size,

                  int write_flag,

                  void *opaque,

                  int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),

                  int (*write_packet)(void *opaque, uint8_t *buf, int buf_size),

                  int64_t (*seek)(void *opaque, int64_t offset, int whence))

{

    s->buffer = buffer;

    s->buffer_size = buffer_size;

    s->buf_ptr = buffer;

    url_resetbuf(s, write_flag ? URL_WRONLY : URL_RDONLY);

    s->opaque = opaque;

    s->write_packet = write_packet;

    s->read_packet = read_packet;

    s->seek = seek;

    s->pos = 0;

    s->must_flush = 0;

    s->eof_reached = 0;

    s->error = 0;

    s->is_streamed = 0;

    s->max_packet_size = 0;

    s->update_checksum= NULL;

    if(!read_packet && !write_flag){

        s->pos = buffer_size;

        s->buf_end = s->buffer + buffer_size;

    }

    s->read_pause = NULL;

    s->read_seek  = NULL;

    return 0;

}
