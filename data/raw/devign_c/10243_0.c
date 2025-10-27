static void configure_alarms(char const *opt)

{

    int i;

    int cur = 0;

    int count = (sizeof(alarm_timers) / sizeof(*alarm_timers)) - 1;

    char *arg;

    char *name;



    if (!strcmp(opt, "help")) {

        show_available_alarms();

        exit(0);

    }



    arg = strdup(opt);



    /* Reorder the array */

    name = strtok(arg, ",");

    while (name) {

        struct qemu_alarm_timer tmp;



        for (i = 0; i < count; i++) {

            if (!strcmp(alarm_timers[i].name, name))

                break;

        }



        if (i == count) {

            fprintf(stderr, "Unknown clock %s\n", name);

            goto next;

        }



        if (i < cur)

            /* Ignore */

            goto next;



	/* Swap */

        tmp = alarm_timers[i];

        alarm_timers[i] = alarm_timers[cur];

        alarm_timers[cur] = tmp;



        cur++;

next:

        name = strtok(NULL, ",");

    }



    free(arg);



    if (cur) {

	/* Disable remaining timers */

        for (i = cur; i < count; i++)

            alarm_timers[i].name = NULL;

    }



    /* debug */

    show_available_alarms();

}
