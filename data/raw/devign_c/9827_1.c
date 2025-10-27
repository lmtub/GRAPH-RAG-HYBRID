static void tcg_handle_interrupt(CPUState *cpu, int mask)

{

    int old_mask;



    old_mask = cpu->interrupt_request;

    cpu->interrupt_request |= mask;



    /*

     * If called from iothread context, wake the target cpu in

     * case its halted.

     */

    if (!qemu_cpu_is_self(cpu)) {

        qemu_cpu_kick(cpu);

        return;

    }



    if (use_icount) {

        cpu->icount_decr.u16.high = 0xffff;

        if (!cpu->can_do_io

            && (mask & ~old_mask) != 0) {

            cpu_abort(cpu, "Raised interrupt while not in I/O function");

        }

    } else {

        cpu->tcg_exit_req = 1;

    }

}
