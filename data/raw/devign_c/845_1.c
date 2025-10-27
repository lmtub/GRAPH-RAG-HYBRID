static void do_interrupt_user(CPUX86State *env, int intno, int is_int,

                              int error_code, target_ulong next_eip)

{

    SegmentCache *dt;

    target_ulong ptr;

    int dpl, cpl, shift;

    uint32_t e2;



    dt = &env->idt;

    if (env->hflags & HF_LMA_MASK) {

        shift = 4;

    } else {

        shift = 3;

    }

    ptr = dt->base + (intno << shift);

    e2 = cpu_ldl_kernel(env, ptr + 4);



    dpl = (e2 >> DESC_DPL_SHIFT) & 3;

    cpl = env->hflags & HF_CPL_MASK;

    /* check privilege if software int */

    if (is_int && dpl < cpl) {

        raise_exception_err(env, EXCP0D_GPF, (intno << shift) + 2);

    }



    /* Since we emulate only user space, we cannot do more than

       exiting the emulation with the suitable exception and error

       code. So update EIP for INT 0x80 and EXCP_SYSCALL. */

    if (is_int || intno == EXCP_SYSCALL) {

        env->eip = next_eip;

    }

}
