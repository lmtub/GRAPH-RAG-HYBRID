static void wait_for_serial(const char *side)

{

    char *serialpath = g_strdup_printf("%s/%s", tmpfs, side);

    FILE *serialfile = fopen(serialpath, "r");

    const char *arch = qtest_get_arch();

    int started = (strcmp(side, "src_serial") == 0 &&

                   strcmp(arch, "ppc64") == 0) ? 0 : 1;



    do {

        int readvalue = fgetc(serialfile);



        if (!started) {

            /* SLOF prints its banner before starting test,

             * to ignore it, mark the start of the test with '_',

             * ignore all characters until this marker

             */

            switch (readvalue) {

            case '_':

                started = 1;

                break;

            case EOF:

                fseek(serialfile, 0, SEEK_SET);

                usleep(1000);

                break;

            }

            continue;

        }

        switch (readvalue) {

        case 'A':

            /* Fine */

            break;



        case 'B':

            /* It's alive! */

            fclose(serialfile);

            g_free(serialpath);

            return;



        case EOF:

            started = (strcmp(side, "src_serial") == 0 &&

                       strcmp(arch, "ppc64") == 0) ? 0 : 1;

            fseek(serialfile, 0, SEEK_SET);

            usleep(1000);

            break;



        default:

            fprintf(stderr, "Unexpected %d on %s serial\n", readvalue, side);

            g_assert_not_reached();

        }

    } while (true);

}
