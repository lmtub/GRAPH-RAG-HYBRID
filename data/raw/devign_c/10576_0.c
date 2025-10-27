static int open_self_cmdline(void *cpu_env, int fd)

{

    int fd_orig = -1;

    bool word_skipped = false;



    fd_orig = open("/proc/self/cmdline", O_RDONLY);

    if (fd_orig < 0) {

        return fd_orig;

    }



    while (true) {

        ssize_t nb_read;

        char buf[128];

        char *cp_buf = buf;



        nb_read = read(fd_orig, buf, sizeof(buf));

        if (nb_read < 0) {

            int e = errno;

            fd_orig = close(fd_orig);

            errno = e;

            return -1;

        } else if (nb_read == 0) {

            break;

        }



        if (!word_skipped) {

            /* Skip the first string, which is the path to qemu-*-static

               instead of the actual command. */

            cp_buf = memchr(buf, 0, sizeof(buf));

            if (cp_buf) {

                /* Null byte found, skip one string */

                cp_buf++;

                nb_read -= cp_buf - buf;

                word_skipped = true;

            }

        }



        if (word_skipped) {

            if (write(fd, cp_buf, nb_read) != nb_read) {

                int e = errno;

                close(fd_orig);

                errno = e;

                return -1;

            }

        }

    }



    return close(fd_orig);

}
