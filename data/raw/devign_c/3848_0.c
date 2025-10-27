static void cpu_unlink_tb(CPUState *env)

{

#if defined(CONFIG_USE_NPTL)

    /* FIXME: TB unchaining isn't SMP safe.  For now just ignore the

       problem and hope the cpu will stop of its own accord.  For userspace

       emulation this often isn't actually as bad as it sounds.  Often

       signals are used primarily to interrupt blocking syscalls.  */

#else

    TranslationBlock *tb;

    static spinlock_t interrupt_lock = SPIN_LOCK_UNLOCKED;



    tb = env->current_tb;

    /* if the cpu is currently executing code, we must unlink it and

       all the potentially executing TB */

    if (tb && !testandset(&interrupt_lock)) {

        env->current_tb = NULL;

        tb_reset_jump_recursive(tb);

        resetlock(&interrupt_lock);

    }

#endif

}
