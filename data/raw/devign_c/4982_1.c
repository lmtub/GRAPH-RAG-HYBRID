void qemu_set_log(int log_flags, bool use_own_buffers)

{

    qemu_loglevel = log_flags;

    if (qemu_loglevel && !qemu_logfile) {

        qemu_logfile = fopen(logfilename, log_append ? "a" : "w");

        if (!qemu_logfile) {

            perror(logfilename);

            _exit(1);

        }

        /* must avoid mmap() usage of glibc by setting a buffer "by hand" */

        if (use_own_buffers) {

            static char logfile_buf[4096];



            setvbuf(qemu_logfile, logfile_buf, _IOLBF, sizeof(logfile_buf));

        } else {

#if defined(_WIN32)

            /* Win32 doesn't support line-buffering, so use unbuffered output. */

            setvbuf(qemu_logfile, NULL, _IONBF, 0);

#else

            setvbuf(qemu_logfile, NULL, _IOLBF, 0);

#endif

            log_append = 1;

        }

    }

    if (!qemu_loglevel && qemu_logfile) {

        fclose(qemu_logfile);

        qemu_logfile = NULL;

    }

}
