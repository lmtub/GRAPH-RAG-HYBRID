static int net_bridge_run_helper(const char *helper, const char *bridge)

{

    sigset_t oldmask, mask;

    int pid, status;

    char *args[5];

    char **parg;

    int sv[2];



    sigemptyset(&mask);

    sigaddset(&mask, SIGCHLD);

    sigprocmask(SIG_BLOCK, &mask, &oldmask);



    if (socketpair(PF_UNIX, SOCK_STREAM, 0, sv) == -1) {

        return -1;

    }



    /* try to launch bridge helper */

    pid = fork();

    if (pid == 0) {

        int open_max = sysconf(_SC_OPEN_MAX), i;

        char fd_buf[6+10];

        char br_buf[6+IFNAMSIZ] = {0};

        char helper_cmd[PATH_MAX + sizeof(fd_buf) + sizeof(br_buf) + 15];



        for (i = 3; i < open_max; i++) {

            if (i != sv[1]) {

                close(i);

            }

        }



        snprintf(fd_buf, sizeof(fd_buf), "%s%d", "--fd=", sv[1]);



        if (strrchr(helper, ' ') || strrchr(helper, '\t')) {

            /* assume helper is a command */



            if (strstr(helper, "--br=") == NULL) {

                snprintf(br_buf, sizeof(br_buf), "%s%s", "--br=", bridge);

            }



            snprintf(helper_cmd, sizeof(helper_cmd), "%s %s %s %s",

                     helper, "--use-vnet", fd_buf, br_buf);



            parg = args;

            *parg++ = (char *)"sh";

            *parg++ = (char *)"-c";

            *parg++ = helper_cmd;

            *parg++ = NULL;



            execv("/bin/sh", args);

        } else {

            /* assume helper is just the executable path name */



            snprintf(br_buf, sizeof(br_buf), "%s%s", "--br=", bridge);



            parg = args;

            *parg++ = (char *)helper;

            *parg++ = (char *)"--use-vnet";

            *parg++ = fd_buf;

            *parg++ = br_buf;

            *parg++ = NULL;



            execv(helper, args);

        }

        _exit(1);



    } else if (pid > 0) {

        int fd;



        close(sv[1]);



        do {

            fd = recv_fd(sv[0]);

        } while (fd == -1 && errno == EINTR);



        close(sv[0]);



        while (waitpid(pid, &status, 0) != pid) {

            /* loop */

        }

        sigprocmask(SIG_SETMASK, &oldmask, NULL);

        if (fd < 0) {

            fprintf(stderr, "failed to recv file descriptor\n");

            return -1;

        }



        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {

            return fd;

        }

    }

    fprintf(stderr, "failed to launch bridge helper\n");

    return -1;

}
