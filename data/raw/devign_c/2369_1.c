static void start_tcg_kick_timer(void)

{

    if (!tcg_kick_vcpu_timer && CPU_NEXT(first_cpu)) {

        tcg_kick_vcpu_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,

                                           kick_tcg_thread, NULL);

        timer_mod(tcg_kick_vcpu_timer, qemu_tcg_next_kick());

    }

}
