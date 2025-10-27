void cpu_interrupt(CPUState *env, int mask)

{

    int old_mask;



    old_mask = env->interrupt_request;

    env->interrupt_request |= mask;



#ifndef CONFIG_USER_ONLY

    /*

     * If called from iothread context, wake the target cpu in

     * case its halted.

     */

    if (!qemu_cpu_self(env)) {

        qemu_cpu_kick(env);

        return;

    }

#endif



    if (use_icount) {

        env->icount_decr.u16.high = 0xffff;

#ifndef CONFIG_USER_ONLY

        if (!can_do_io(env)

            && (mask & ~old_mask) != 0) {

            cpu_abort(env, "Raised interrupt while not in I/O function");

        }

#endif

    } else {

        cpu_unlink_tb(env);

    }

}
