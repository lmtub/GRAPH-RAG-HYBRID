Coroutine *qemu_coroutine_new(void)

{

    const size_t stack_size = 1 << 20;

    CoroutineUContext *co;

    CoroutineThreadState *coTS;

    struct sigaction sa;

    struct sigaction osa;

    stack_t ss;

    stack_t oss;

    sigset_t sigs;

    sigset_t osigs;

    jmp_buf old_env;



    /* The way to manipulate stack is with the sigaltstack function. We

     * prepare a stack, with it delivering a signal to ourselves and then

     * put sigsetjmp/siglongjmp where needed.

     * This has been done keeping coroutine-ucontext as a model and with the

     * pth ideas (GNU Portable Threads). See coroutine-ucontext for the basics

     * of the coroutines and see pth_mctx.c (from the pth project) for the

     * sigaltstack way of manipulating stacks.

     */



    co = g_malloc0(sizeof(*co));

    co->stack = g_malloc(stack_size);

    co->base.entry_arg = &old_env; /* stash away our jmp_buf */



    coTS = coroutine_get_thread_state();

    coTS->tr_handler = co;



    /*

     * Preserve the SIGUSR2 signal state, block SIGUSR2,

     * and establish our signal handler. The signal will

     * later transfer control onto the signal stack.

     */

    sigemptyset(&sigs);

    sigaddset(&sigs, SIGUSR2);

    pthread_sigmask(SIG_BLOCK, &sigs, &osigs);

    sa.sa_handler = coroutine_trampoline;

    sigfillset(&sa.sa_mask);

    sa.sa_flags = SA_ONSTACK;

    if (sigaction(SIGUSR2, &sa, &osa) != 0) {

        abort();

    }



    /*

     * Set the new stack.

     */

    ss.ss_sp = co->stack;

    ss.ss_size = stack_size;

    ss.ss_flags = 0;

    if (sigaltstack(&ss, &oss) < 0) {

        abort();

    }



    /*

     * Now transfer control onto the signal stack and set it up.

     * It will return immediately via "return" after the sigsetjmp()

     * was performed. Be careful here with race conditions.  The

     * signal can be delivered the first time sigsuspend() is

     * called.

     */

    coTS->tr_called = 0;

    pthread_kill(pthread_self(), SIGUSR2);

    sigfillset(&sigs);

    sigdelset(&sigs, SIGUSR2);

    while (!coTS->tr_called) {

        sigsuspend(&sigs);

    }



    /*

     * Inform the system that we are back off the signal stack by

     * removing the alternative signal stack. Be careful here: It

     * first has to be disabled, before it can be removed.

     */

    sigaltstack(NULL, &ss);

    ss.ss_flags = SS_DISABLE;

    if (sigaltstack(&ss, NULL) < 0) {

        abort();

    }

    sigaltstack(NULL, &ss);

    if (!(oss.ss_flags & SS_DISABLE)) {

        sigaltstack(&oss, NULL);

    }



    /*

     * Restore the old SIGUSR2 signal handler and mask

     */

    sigaction(SIGUSR2, &osa, NULL);

    pthread_sigmask(SIG_SETMASK, &osigs, NULL);



    /*

     * Now enter the trampoline again, but this time not as a signal

     * handler. Instead we jump into it directly. The functionally

     * redundant ping-pong pointer arithmetic is necessary to avoid

     * type-conversion warnings related to the `volatile' qualifier and

     * the fact that `jmp_buf' usually is an array type.

     */

    if (!sigsetjmp(old_env, 0)) {

        siglongjmp(coTS->tr_reenter, 1);

    }



    /*

     * Ok, we returned again, so now we're finished

     */



    return &co->base;

}
