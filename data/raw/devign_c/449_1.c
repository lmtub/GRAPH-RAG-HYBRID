void configure_alarms(char const *opt)

{

    int i;

    int cur = 0;

    int count = ARRAY_SIZE(alarm_timers) - 1;

    char *arg;

    char *name;

    struct qemu_alarm_timer tmp;



    if (!strcmp(opt, "?")) {

        show_available_alarms();

        exit(0);

    }



    arg = g_strdup(opt);



    /* Reorder the array */

    name = strtok(arg, ",");

    while (name) {

        for (i = 0; i < count && alarm_timers[i].name; i++) {

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



    g_free(arg);



    if (cur) {

        /* Disable remaining timers */

        for (i = cur; i < count; i++)

            alarm_timers[i].name = NULL;

    } else {

        show_available_alarms();

        exit(1);

    }

}
