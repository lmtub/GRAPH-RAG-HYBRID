void qmp_dump_guest_memory(bool paging, const char *file, bool has_begin,

                           int64_t begin, bool has_length, int64_t length,

                           Error **errp)

{

    const char *p;

    int fd = -1;

    DumpState *s;

    int ret;



    if (has_begin && !has_length) {

        error_set(errp, QERR_MISSING_PARAMETER, "length");

        return;

    }

    if (!has_begin && has_length) {

        error_set(errp, QERR_MISSING_PARAMETER, "begin");

        return;

    }



#if !defined(WIN32)

    if (strstart(file, "fd:", &p)) {

        fd = monitor_get_fd(cur_mon, p, errp);

        if (fd == -1) {

            return;

        }

    }

#endif



    if  (strstart(file, "file:", &p)) {

        fd = qemu_open(p, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, S_IRUSR);

        if (fd < 0) {

            error_setg_file_open(errp, errno, p);

            return;

        }

    }



    if (fd == -1) {

        error_set(errp, QERR_INVALID_PARAMETER, "protocol");

        return;

    }



    s = g_malloc(sizeof(DumpState));



    ret = dump_init(s, fd, paging, has_begin, begin, length, errp);

    if (ret < 0) {

        g_free(s);

        return;

    }



    if (create_vmcore(s) < 0 && !error_is_set(s->errp)) {

        error_set(errp, QERR_IO_ERROR);

    }



    g_free(s);

}
