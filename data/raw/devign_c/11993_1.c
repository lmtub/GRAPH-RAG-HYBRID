static void setup_frame_v1(int usig, struct target_sigaction *ka,

                           target_sigset_t *set, CPUARMState *regs)

{

	struct sigframe_v1 *frame;

	abi_ulong frame_addr = get_sigframe(ka, regs, sizeof(*frame));

	int i;



	if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

		return;



	setup_sigcontext(&frame->sc, regs, set->sig[0]);



        for(i = 1; i < TARGET_NSIG_WORDS; i++) {

            if (__put_user(set->sig[i], &frame->extramask[i - 1]))

                goto end;

	}



        setup_return(regs, ka, &frame->retcode, frame_addr, usig,

                     frame_addr + offsetof(struct sigframe_v1, retcode));



end:

	unlock_user_struct(frame, frame_addr, 1);

}
