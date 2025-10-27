static int64_t cpu_get_icount_locked(void)

{

    int64_t icount;

    CPUState *cpu = current_cpu;



    icount = timers_state.qemu_icount;

    if (cpu) {

        if (!cpu_can_do_io(cpu)) {

            fprintf(stderr, "Bad clock read\n");

        }

        icount -= (cpu->icount_decr.u16.low + cpu->icount_extra);

    }

    return timers_state.qemu_icount_bias + cpu_icount_to_ns(icount);

}
