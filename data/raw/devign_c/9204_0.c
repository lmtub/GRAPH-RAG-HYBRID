static void monitor_print_error(Monitor *mon)

{

    qerror_print(mon->error);

    QDECREF(mon->error);

    mon->error = NULL;

}
