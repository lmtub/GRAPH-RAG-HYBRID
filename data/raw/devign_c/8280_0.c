static void setup_frame(int usig, struct emulated_sigaction *ka,

			target_sigset_t *set, CPUState *regs)

{

	struct sigframe *frame;

	abi_ulong frame_addr = get_sigframe(ka, regs, sizeof(*frame));

	int i, err = 0;



	if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

		return;



	err |= setup_sigcontext(&frame->sc, /*&frame->fpstate,*/ regs, set->sig[0]);



        for(i = 1; i < TARGET_NSIG_WORDS; i++) {

            if (__put_user(set->sig[i], &frame->extramask[i - 1]))

                goto end;

	}



	if (err == 0)

            err = setup_return(regs, ka, &frame->retcode, frame, usig);



end:

	unlock_user_struct(frame, frame_addr, 1);

        //	return err;

}
