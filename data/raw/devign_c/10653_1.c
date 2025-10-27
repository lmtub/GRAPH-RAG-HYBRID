static void qemu_clock_init(QEMUClockType type)
{
    QEMUClock *clock = qemu_clock_ptr(type);
    clock->type = type;
    clock->enabled = true;
    clock->last = INT64_MIN;
    QLIST_INIT(&clock->timerlists);
    notifier_list_init(&clock->reset_notifiers);
    main_loop_tlg.tl[type] = timerlist_new(type, NULL, NULL);
}