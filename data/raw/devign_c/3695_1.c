int ff_win32_open(const char *filename_utf8, int oflag, int pmode)

{

    int fd;

    int num_chars;

    wchar_t *filename_w;



    /* convert UTF-8 to wide chars */

    num_chars = MultiByteToWideChar(CP_UTF8, 0, filename_utf8, -1, NULL, 0);

    if (num_chars <= 0)

        return -1;

    filename_w = av_mallocz(sizeof(wchar_t) * num_chars);

    MultiByteToWideChar(CP_UTF8, 0, filename_utf8, -1, filename_w, num_chars);



    fd = _wsopen(filename_w, oflag, SH_DENYNO, pmode);

    av_freep(&filename_w);



    /* filename maybe be in CP_ACP */

    if (fd == -1 && !(oflag & O_CREAT))

        return _sopen(filename_utf8, oflag, SH_DENYNO, pmode);



    return fd;

}
