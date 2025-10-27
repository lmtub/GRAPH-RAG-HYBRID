static QEMUClock *qemu_new_clock(int type)

{

    QEMUClock *clock;



    clock = g_malloc0(sizeof(QEMUClock));

    clock->type = type;

    clock->enabled = true;

    clock->last = INT64_MIN;

    notifier_list_init(&clock->reset_notifiers);

    return clock;

}
