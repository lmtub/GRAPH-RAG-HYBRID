static void setup_sigframe_v2(struct target_ucontext_v2 *uc,

                              target_sigset_t *set, CPUState *env)

{

    struct target_sigaltstack stack;

    int i;



    /* Clear all the bits of the ucontext we don't use.  */

    memset(uc, 0, offsetof(struct target_ucontext_v2, tuc_mcontext));



    memset(&stack, 0, sizeof(stack));

    __put_user(target_sigaltstack_used.ss_sp, &stack.ss_sp);

    __put_user(target_sigaltstack_used.ss_size, &stack.ss_size);

    __put_user(sas_ss_flags(get_sp_from_cpustate(env)), &stack.ss_flags);

    memcpy(&uc->tuc_stack, &stack, sizeof(stack));



    setup_sigcontext(&uc->tuc_mcontext, env, set->sig[0]);

    /* FIXME: Save coprocessor signal frame.  */

    for(i = 0; i < TARGET_NSIG_WORDS; i++) {

        __put_user(set->sig[i], &uc->tuc_sigmask.sig[i]);

    }

}
