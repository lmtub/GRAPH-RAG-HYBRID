static int stdio_get_buffer(void *opaque, uint8_t *buf, int64_t pos, int size)

{

    QEMUFileStdio *s = opaque;

    FILE *fp = s->stdio_file;

    int bytes;



    do {

        clearerr(fp);

        bytes = fread(buf, 1, size, fp);

    } while ((bytes == 0) && ferror(fp) && (errno == EINTR));

    return bytes;

}
