void init_clocks(void)

{

    QEMUClockType type;

    for (type = 0; type < QEMU_CLOCK_MAX; type++) {

        qemu_clock_init(type);

    }



#ifdef CONFIG_PRCTL_PR_SET_TIMERSLACK

    prctl(PR_SET_TIMERSLACK, 1, 0, 0, 0);

#endif

}
