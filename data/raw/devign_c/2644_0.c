static int file_open(URLContext *h, const char *filename, int flags)

{

    int access;

    int fd;



    av_strstart(filename, "file:", &filename);



    if (flags & URL_RDWR) {

        access = O_CREAT | O_TRUNC | O_RDWR;

    } else if (flags & URL_WRONLY) {

        access = O_CREAT | O_TRUNC | O_WRONLY;

    } else {

        access = O_RDONLY;

    }

#if defined(__MINGW32__) || defined(CONFIG_OS2) || defined(__CYGWIN__)

    access |= O_BINARY;

#endif

    fd = open(filename, access, 0666);

    if (fd < 0)

        return AVERROR(ENOENT);

    h->priv_data = (void *)(size_t)fd;

    return 0;

}
