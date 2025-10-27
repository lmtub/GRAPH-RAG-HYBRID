long do_sigreturn(CPUMIPSState *regs)

{

    struct sigframe *frame;

    abi_ulong frame_addr;

    sigset_t blocked;

    target_sigset_t target_set;

    int i;



#if defined(DEBUG_SIGNAL)

    fprintf(stderr, "do_sigreturn\n");

#endif

    frame_addr = regs->active_tc.gpr[29];

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1))

   	goto badframe;



    for(i = 0; i < TARGET_NSIG_WORDS; i++) {

   	if(__get_user(target_set.sig[i], &frame->sf_mask.sig[i]))

	    goto badframe;

    }



    target_to_host_sigset_internal(&blocked, &target_set);

    do_sigprocmask(SIG_SETMASK, &blocked, NULL);



    restore_sigcontext(regs, &frame->sf_sc);



#if 0

    /*

     * Don't let your children do this ...

     */

    __asm__ __volatile__(

   	"move\t$29, %0\n\t"

   	"j\tsyscall_exit"

   	:/* no outputs */

   	:"r" (&regs));

    /* Unreached */

#endif



    regs->active_tc.PC = regs->CP0_EPC;

    mips_set_hflags_isa_mode_from_pc(regs);

    /* I am not sure this is right, but it seems to work

    * maybe a problem with nested signals ? */

    regs->CP0_EPC = 0;

    return -TARGET_QEMU_ESIGRETURN;



badframe:

    force_sig(TARGET_SIGSEGV/*, current*/);

    return 0;

}
