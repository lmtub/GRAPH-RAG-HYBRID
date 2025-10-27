AVIOContext *avio_alloc_context(

                  unsigned char *buffer,

                  int buffer_size,

                  int write_flag,

                  void *opaque,

                  int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),

                  int (*write_packet)(void *opaque, uint8_t *buf, int buf_size),

                  int64_t (*seek)(void *opaque, int64_t offset, int whence))

{

    AVIOContext *s = av_mallocz(sizeof(AVIOContext));



    ffio_init_context(s, buffer, buffer_size, write_flag, opaque,

                  read_packet, write_packet, seek);

    return s;

}