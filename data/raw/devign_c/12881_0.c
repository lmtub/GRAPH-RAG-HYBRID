static void file_put_buffer(void *opaque, const uint8_t *buf,

                            int64_t pos, int size)

{

    QEMUFileStdio *s = opaque;

    fseek(s->outfile, pos, SEEK_SET);

    fwrite(buf, 1, size, s->outfile);

}
