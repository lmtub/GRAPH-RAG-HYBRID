static void *sigwait_compat(void *opaque)

{

    struct sigfd_compat_info *info = opaque;

    int err;

    sigset_t all;



    sigfillset(&all);

    sigprocmask(SIG_BLOCK, &all, NULL);



    do {

        siginfo_t siginfo;



        err = sigwaitinfo(&info->mask, &siginfo);

        if (err == -1 && errno == EINTR) {

            err = 0;

            continue;

        }



        if (err > 0) {

            char buffer[128];

            size_t offset = 0;



            memcpy(buffer, &err, sizeof(err));

            while (offset < sizeof(buffer)) {

                ssize_t len;



                len = write(info->fd, buffer + offset,

                            sizeof(buffer) - offset);

                if (len == -1 && errno == EINTR)

                    continue;



                if (len <= 0) {

                    err = -1;

                    break;

                }



                offset += len;

            }

        }

    } while (err >= 0);



    return NULL;

}
