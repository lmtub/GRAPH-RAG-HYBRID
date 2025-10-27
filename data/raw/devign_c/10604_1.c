static void check_guest_output(const testdef_t *test, int fd)

{

    bool output_ok = false;

    int i, nbr, pos = 0;

    char ch;



    /* Poll serial output... Wait at most 60 seconds */

    for (i = 0; i < 6000; ++i) {

        while ((nbr = read(fd, &ch, 1)) == 1) {

            if (ch == test->expect[pos]) {

                pos += 1;

                if (test->expect[pos] == '\0') {

                    /* We've reached the end of the expected string! */

                    output_ok = true;

                    goto done;

                }

            } else {

                pos = 0;

            }

        }

        g_assert(nbr >= 0);

        g_usleep(10000);

    }



done:

    g_assert(output_ok);

}
