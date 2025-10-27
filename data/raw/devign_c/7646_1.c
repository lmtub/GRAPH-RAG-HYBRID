static int stdio_get_fd(void *opaque)

{

    QEMUFileStdio *s = opaque;



    return fileno(s->stdio_file);

}
