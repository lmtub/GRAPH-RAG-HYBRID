int block_signals(void)

{

    TaskState *ts = (TaskState *)thread_cpu->opaque;

    sigset_t set;

    int pending;



    /* It's OK to block everything including SIGSEGV, because we won't

     * run any further guest code before unblocking signals in

     * process_pending_signals().

     */

    sigfillset(&set);

    sigprocmask(SIG_SETMASK, &set, 0);



    pending = atomic_xchg(&ts->signal_pending, 1);



    return pending;

}
