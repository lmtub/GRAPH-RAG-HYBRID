int av_file_map(const char *filename, uint8_t **bufptr, size_t *size,

                int log_offset, void *log_ctx)

{

    FileLogContext file_log_ctx = { &file_log_ctx_class, log_offset, log_ctx };

    int err, fd = open(filename, O_RDONLY);

    struct stat st;

    av_unused void *ptr;

    off_t off_size;

    char errbuf[128];

    *bufptr = NULL;



    if (fd < 0) {

        err = AVERROR(errno);

        av_strerror(err, errbuf, sizeof(errbuf));

        av_log(&file_log_ctx, AV_LOG_ERROR, "Cannot read file '%s': %s\n", filename, errbuf);

        return err;

    }



    if (fstat(fd, &st) < 0) {

        err = AVERROR(errno);

        av_strerror(err, errbuf, sizeof(errbuf));

        av_log(&file_log_ctx, AV_LOG_ERROR, "Error occurred in fstat(): %s\n", errbuf);

        close(fd);

        return err;

    }



    off_size = st.st_size;

    if (off_size > SIZE_MAX) {

        av_log(&file_log_ctx, AV_LOG_ERROR,

               "File size for file '%s' is too big\n", filename);

        close(fd);

        return AVERROR(EINVAL);

    }

    *size = off_size;



#if HAVE_MMAP

    ptr = mmap(NULL, *size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);

    if ((int)(ptr) == -1) {

        err = AVERROR(errno);

        av_strerror(err, errbuf, sizeof(errbuf));

        av_log(&file_log_ctx, AV_LOG_ERROR, "Error occurred in mmap(): %s\n", errbuf);

        close(fd);

        return err;

    }

    *bufptr = ptr;

#elif HAVE_MAPVIEWOFFILE

    {

        HANDLE mh, fh = (HANDLE)_get_osfhandle(fd);



        mh = CreateFileMapping(fh, NULL, PAGE_READONLY, 0, 0, NULL);

        if (!mh) {

            av_log(&file_log_ctx, AV_LOG_ERROR, "Error occurred in CreateFileMapping()\n");

            close(fd);

            return -1;

        }



        ptr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, *size);

        CloseHandle(mh);

        if (!ptr) {

            av_log(&file_log_ctx, AV_LOG_ERROR, "Error occurred in MapViewOfFile()\n");

            close(fd);

            return -1;

        }



        *bufptr = ptr;

    }

#else

    *bufptr = av_malloc(*size);

    if (!*bufptr) {

        av_log(&file_log_ctx, AV_LOG_ERROR, "Memory allocation error occurred\n");

        close(fd);

        return AVERROR(ENOMEM);

    }

    read(fd, *bufptr, *size);

#endif



    close(fd);

    return 0;

}
