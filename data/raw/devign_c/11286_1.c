void cpu_interrupt(CPUState *env, int mask)

{

#if !defined(USE_NPTL)

    TranslationBlock *tb;

    static spinlock_t interrupt_lock = SPIN_LOCK_UNLOCKED;

#endif

    int old_mask;



    old_mask = env->interrupt_request;

    /* FIXME: This is probably not threadsafe.  A different thread could

       be in the middle of a read-modify-write operation.  */

    env->interrupt_request |= mask;

#if defined(USE_NPTL)

    /* FIXME: TB unchaining isn't SMP safe.  For now just ignore the

       problem and hope the cpu will stop of its own accord.  For userspace

       emulation this often isn't actually as bad as it sounds.  Often

       signals are used primarily to interrupt blocking syscalls.  */

#else

    if (use_icount) {

        env->icount_decr.u16.high = 0xffff;

#ifndef CONFIG_USER_ONLY

        /* CPU_INTERRUPT_EXIT isn't a real interrupt.  It just means

           an async event happened and we need to process it.  */

        if (!can_do_io(env)

            && (mask & ~(old_mask | CPU_INTERRUPT_EXIT)) != 0) {

            cpu_abort(env, "Raised interrupt while not in I/O function");

        }

#endif

    } else {

        tb = env->current_tb;

        /* if the cpu is currently executing code, we must unlink it and

           all the potentially executing TB */

        if (tb && !testandset(&interrupt_lock)) {

            env->current_tb = NULL;

            tb_reset_jump_recursive(tb);

            resetlock(&interrupt_lock);

        }

    }

#endif

}
