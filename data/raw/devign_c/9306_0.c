void qemu_register_clock_reset_notifier(QEMUClock *clock,

                                        Notifier *notifier)

{

    qemu_clock_register_reset_notifier(clock->type, notifier);

}
