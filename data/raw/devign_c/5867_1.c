int select_watchdog(const char *p)

{

    WatchdogTimerModel *model;



    if (watchdog) {

        fprintf(stderr,

                 "qemu: only one watchdog option may be given\n");

        return 1;

    }



    /* -watchdog ? lists available devices and exits cleanly. */

    if (strcmp(p, "?") == 0) {

        LIST_FOREACH(model, &watchdog_list, entry) {

            fprintf(stderr, "\t%s\t%s\n",

                     model->wdt_name, model->wdt_description);

        }

        return 2;

    }



    LIST_FOREACH(model, &watchdog_list, entry) {

        if (strcasecmp(model->wdt_name, p) == 0) {

            watchdog = model;

            return 0;

        }

    }



    fprintf(stderr, "Unknown -watchdog device. Supported devices are:\n");

    LIST_FOREACH(model, &watchdog_list, entry) {

        fprintf(stderr, "\t%s\t%s\n",

                 model->wdt_name, model->wdt_description);

    }

    return 1;

}
