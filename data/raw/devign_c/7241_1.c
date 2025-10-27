static int stdio_put_buffer(void *opaque, const uint8_t *buf, int64_t pos,

                            int size)

{

    QEMUFileStdio *s = opaque;

    return fwrite(buf, 1, size, s->stdio_file);

}
