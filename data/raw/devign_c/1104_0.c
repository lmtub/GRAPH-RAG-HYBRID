static int configure_accelerator(void)

{

    const char *p = NULL;

    char buf[10];

    int i, ret;

    bool accel_initalised = 0;

    bool init_failed = 0;



    QemuOptsList *list = qemu_find_opts("machine");

    if (!QTAILQ_EMPTY(&list->head)) {

        p = qemu_opt_get(QTAILQ_FIRST(&list->head), "accel");

    }



    if (p == NULL) {

        /* Use the default "accelerator", tcg */

        p = "tcg";

    }



    while (!accel_initalised && *p != '\0') {

        if (*p == ':') {

            p++;

        }

        p = get_opt_name(buf, sizeof (buf), p, ':');

        for (i = 0; i < ARRAY_SIZE(accel_list); i++) {

            if (strcmp(accel_list[i].opt_name, buf) == 0) {

                ret = accel_list[i].init();

                if (ret < 0) {

                    init_failed = 1;

                    if (!accel_list[i].available()) {

                        printf("%s not supported for this target\n",

                               accel_list[i].name);

                    } else {

                        fprintf(stderr, "failed to initialize %s: %s\n",

                                accel_list[i].name,

                                strerror(-ret));

                    }

                } else {

                    accel_initalised = 1;

                    *(accel_list[i].allowed) = 1;

                }

                break;

            }

        }

        if (i == ARRAY_SIZE(accel_list)) {

            fprintf(stderr, "\"%s\" accelerator does not exist.\n", buf);

        }

    }



    if (!accel_initalised) {

        fprintf(stderr, "No accelerator found!\n");

        exit(1);

    }



    if (init_failed) {

        fprintf(stderr, "Back to %s accelerator.\n", accel_list[i].name);

    }



    return !accel_initalised;

}
