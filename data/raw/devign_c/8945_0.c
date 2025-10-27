long do_sigreturn(CPUPPCState *env)

{

    struct target_sigcontext *sc = NULL;

    struct target_mcontext *sr = NULL;

    target_ulong sr_addr = 0, sc_addr;

    sigset_t blocked;

    target_sigset_t set;



    sc_addr = env->gpr[1] + SIGNAL_FRAMESIZE;

    if (!lock_user_struct(VERIFY_READ, sc, sc_addr, 1))

        goto sigsegv;



#if defined(TARGET_PPC64)

    set.sig[0] = sc->oldmask + ((long)(sc->_unused[3]) << 32);

#else

    if(__get_user(set.sig[0], &sc->oldmask) ||

       __get_user(set.sig[1], &sc->_unused[3]))

       goto sigsegv;

#endif

    target_to_host_sigset_internal(&blocked, &set);

    sigprocmask(SIG_SETMASK, &blocked, NULL);



    if (__get_user(sr_addr, &sc->regs))

        goto sigsegv;

    if (!lock_user_struct(VERIFY_READ, sr, sr_addr, 1))

        goto sigsegv;

    if (restore_user_regs(env, sr, 1))

        goto sigsegv;



    unlock_user_struct(sr, sr_addr, 1);

    unlock_user_struct(sc, sc_addr, 1);

    return -TARGET_QEMU_ESIGRETURN;



sigsegv:

    unlock_user_struct(sr, sr_addr, 1);

    unlock_user_struct(sc, sc_addr, 1);

    qemu_log("segfaulting from do_sigreturn\n");

    force_sig(TARGET_SIGSEGV);

    return 0;

}
