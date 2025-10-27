void qemu_unregister_clock_reset_notifier(QEMUClock *clock,

                                          Notifier *notifier)

{

    qemu_clock_unregister_reset_notifier(clock->type, notifier);

}
