static long do_sigreturn_v2(CPUARMState *env)

{

    abi_ulong frame_addr;

    struct sigframe_v2 *frame = NULL;



    /*

     * Since we stacked the signal on a 64-bit boundary,

     * then 'sp' should be word aligned here.  If it's

     * not, then the user is trying to mess with us.

     */

    frame_addr = env->regs[13];

    trace_user_do_sigreturn(env, frame_addr);

    if (frame_addr & 7) {

        goto badframe;

    }



    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }



    if (do_sigframe_return_v2(env, frame_addr, &frame->uc)) {

        goto badframe;

    }



    unlock_user_struct(frame, frame_addr, 0);

    return env->regs[0];



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV /* , current */);

    return 0;

}
