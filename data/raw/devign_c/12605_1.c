static void bios_supports_mode(const char *pmutils_bin, const char *pmutils_arg,

                               const char *sysfile_str, Error **err)

{

    pid_t pid;

    ssize_t ret;

    char *pmutils_path;

    int status, pipefds[2];



    if (pipe(pipefds) < 0) {

        error_set(err, QERR_UNDEFINED_ERROR);

        return;

    }



    pmutils_path = g_find_program_in_path(pmutils_bin);



    pid = fork();

    if (!pid) {

        struct sigaction act;



        memset(&act, 0, sizeof(act));

        act.sa_handler = SIG_DFL;

        sigaction(SIGCHLD, &act, NULL);



        setsid();

        close(pipefds[0]);

        reopen_fd_to_null(0);

        reopen_fd_to_null(1);

        reopen_fd_to_null(2);



        pid = fork();

        if (!pid) {

            int fd;

            char buf[32]; /* hopefully big enough */



            if (pmutils_path) {

                execle(pmutils_path, pmutils_bin, pmutils_arg, NULL, environ);

            }



            /*

             * If we get here either pm-utils is not installed or execle() has

             * failed. Let's try the manual method if the caller wants it.

             */



            if (!sysfile_str) {

                _exit(SUSPEND_NOT_SUPPORTED);

            }



            fd = open(LINUX_SYS_STATE_FILE, O_RDONLY);

            if (fd < 0) {

                _exit(SUSPEND_NOT_SUPPORTED);

            }



            ret = read(fd, buf, sizeof(buf)-1);

            if (ret <= 0) {

                _exit(SUSPEND_NOT_SUPPORTED);

            }

            buf[ret] = '\0';



            if (strstr(buf, sysfile_str)) {

                _exit(SUSPEND_SUPPORTED);

            }



            _exit(SUSPEND_NOT_SUPPORTED);

        }



        if (pid > 0) {

            wait(&status);

        } else {

            status = SUSPEND_NOT_SUPPORTED;

        }



        ret = write(pipefds[1], &status, sizeof(status));

        if (ret != sizeof(status)) {

            _exit(EXIT_FAILURE);

        }



        _exit(EXIT_SUCCESS);

    }



    close(pipefds[1]);

    g_free(pmutils_path);



    if (pid < 0) {

        error_set(err, QERR_UNDEFINED_ERROR);

        goto out;

    }



    ret = read(pipefds[0], &status, sizeof(status));

    if (ret == sizeof(status) && WIFEXITED(status) &&

        WEXITSTATUS(status) == SUSPEND_SUPPORTED) {

            goto out;

    }



    error_set(err, QERR_UNSUPPORTED);



out:

    close(pipefds[0]);

}
