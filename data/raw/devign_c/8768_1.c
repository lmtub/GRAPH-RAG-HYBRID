qio_channel_command_new_spawn(const char *const argv[],

                              int flags,

                              Error **errp)

{

    pid_t pid = -1;

    int stdinfd[2] = { -1, -1 };

    int stdoutfd[2] = { -1, -1 };

    int devnull = -1;

    bool stdinnull = false, stdoutnull = false;

    QIOChannelCommand *ioc;



    flags = flags & O_ACCMODE;



    if (flags == O_RDONLY) {

        stdinnull = true;

    }

    if (flags == O_WRONLY) {

        stdoutnull = true;

    }



    if (stdinnull || stdoutnull) {

        devnull = open("/dev/null", O_RDWR);

        if (!devnull) {

            error_setg_errno(errp, errno,

                             "Unable to open /dev/null");

            goto error;

        }

    }



    if ((!stdinnull && pipe(stdinfd) < 0) ||

        (!stdoutnull && pipe(stdoutfd) < 0)) {

        error_setg_errno(errp, errno,

                         "Unable to open pipe");

        goto error;

    }



    pid = qemu_fork(errp);

    if (pid < 0) {

        goto error;

    }



    if (pid == 0) { /* child */

        dup2(stdinnull ? devnull : stdinfd[0], STDIN_FILENO);

        dup2(stdoutnull ? devnull : stdoutfd[1], STDOUT_FILENO);

        /* Leave stderr connected to qemu's stderr */



        if (!stdinnull) {

            close(stdinfd[0]);

            close(stdinfd[1]);

        }

        if (!stdoutnull) {

            close(stdoutfd[0]);

            close(stdoutfd[1]);

        }



        execv(argv[0], (char * const *)argv);

        _exit(1);

    }



    if (!stdinnull) {

        close(stdinfd[0]);

    }

    if (!stdoutnull) {

        close(stdoutfd[1]);

    }



    ioc = qio_channel_command_new_pid(stdinnull ? devnull : stdinfd[1],

                                      stdoutnull ? devnull : stdoutfd[0],

                                      pid);

    trace_qio_channel_command_new_spawn(ioc, argv[0], flags);

    return ioc;



 error:

    if (stdinfd[0] != -1) {

        close(stdinfd[0]);

    }

    if (stdinfd[1] != -1) {

        close(stdinfd[1]);

    }

    if (stdoutfd[0] != -1) {

        close(stdoutfd[0]);

    }

    if (stdoutfd[1] != -1) {

        close(stdoutfd[1]);

    }

    return NULL;

}
