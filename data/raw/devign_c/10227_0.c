static inline void helper_ret_protected(int shift, int is_iret, int addend)

{

    uint32_t sp, new_cs, new_eip, new_eflags, new_esp, new_ss;

    uint32_t new_es, new_ds, new_fs, new_gs;

    uint32_t e1, e2, ss_e1, ss_e2;

    int cpl, dpl, rpl, eflags_mask;

    uint8_t *ssp;

    

    sp = ESP;

    if (!(env->segs[R_SS].flags & DESC_B_MASK))

        sp &= 0xffff;

    ssp = env->segs[R_SS].base + sp;

    if (shift == 1) {

        /* 32 bits */

        if (is_iret)

            new_eflags = ldl_kernel(ssp + 8);

        new_cs = ldl_kernel(ssp + 4) & 0xffff;

        new_eip = ldl_kernel(ssp);

        if (is_iret && (new_eflags & VM_MASK))

            goto return_to_vm86;

    } else {

        /* 16 bits */

        if (is_iret)

            new_eflags = lduw_kernel(ssp + 4);

        new_cs = lduw_kernel(ssp + 2);

        new_eip = lduw_kernel(ssp);

    }

    if ((new_cs & 0xfffc) == 0)

        raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

    if (load_segment(&e1, &e2, new_cs) != 0)

        raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

    if (!(e2 & DESC_S_MASK) ||

        !(e2 & DESC_CS_MASK))

        raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

    cpl = env->hflags & HF_CPL_MASK;

    rpl = new_cs & 3; 

    if (rpl < cpl)

        raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

    dpl = (e2 >> DESC_DPL_SHIFT) & 3;

    if (e2 & DESC_CS_MASK) {

        if (dpl > rpl)

            raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

    } else {

        if (dpl != rpl)

            raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

    }

    if (!(e2 & DESC_P_MASK))

        raise_exception_err(EXCP0B_NOSEG, new_cs & 0xfffc);

    

    if (rpl == cpl) {

        /* return to same priledge level */

        cpu_x86_load_seg_cache(env, R_CS, new_cs, 

                       get_seg_base(e1, e2),

                       get_seg_limit(e1, e2),

                       e2);

        new_esp = sp + (4 << shift) + ((2 * is_iret) << shift) + addend;

    } else {

        /* return to different priviledge level */

        ssp += (4 << shift) + ((2 * is_iret) << shift) + addend;

        if (shift == 1) {

            /* 32 bits */

            new_esp = ldl_kernel(ssp);

            new_ss = ldl_kernel(ssp + 4) & 0xffff;

        } else {

            /* 16 bits */

            new_esp = lduw_kernel(ssp);

            new_ss = lduw_kernel(ssp + 2);

        }

        

        if ((new_ss & 3) != rpl)

            raise_exception_err(EXCP0D_GPF, new_ss & 0xfffc);

        if (load_segment(&ss_e1, &ss_e2, new_ss) != 0)

            raise_exception_err(EXCP0D_GPF, new_ss & 0xfffc);

        if (!(ss_e2 & DESC_S_MASK) ||

            (ss_e2 & DESC_CS_MASK) ||

            !(ss_e2 & DESC_W_MASK))

            raise_exception_err(EXCP0D_GPF, new_ss & 0xfffc);

        dpl = (ss_e2 >> DESC_DPL_SHIFT) & 3;

        if (dpl != rpl)

            raise_exception_err(EXCP0D_GPF, new_ss & 0xfffc);

        if (!(ss_e2 & DESC_P_MASK))

            raise_exception_err(EXCP0B_NOSEG, new_ss & 0xfffc);



        cpu_x86_load_seg_cache(env, R_CS, new_cs, 

                       get_seg_base(e1, e2),

                       get_seg_limit(e1, e2),

                       e2);

        cpu_x86_load_seg_cache(env, R_SS, new_ss, 

                       get_seg_base(ss_e1, ss_e2),

                       get_seg_limit(ss_e1, ss_e2),

                       ss_e2);

        cpu_x86_set_cpl(env, rpl);

    }

    if (env->segs[R_SS].flags & DESC_B_MASK)

        ESP = new_esp;

    else

        ESP = (ESP & 0xffff0000) | 

            (new_esp & 0xffff);

    env->eip = new_eip;

    if (is_iret) {

        /* NOTE: 'cpl' can be different from the current CPL */

        if (cpl == 0)

            eflags_mask = FL_UPDATE_CPL0_MASK;

        else

            eflags_mask = FL_UPDATE_MASK32;

        if (shift == 0)

            eflags_mask &= 0xffff;

        load_eflags(new_eflags, eflags_mask);

    }

    return;



 return_to_vm86:

    new_esp = ldl_kernel(ssp + 12);

    new_ss = ldl_kernel(ssp + 16);

    new_es = ldl_kernel(ssp + 20);

    new_ds = ldl_kernel(ssp + 24);

    new_fs = ldl_kernel(ssp + 28);

    new_gs = ldl_kernel(ssp + 32);

    

    /* modify processor state */

    load_eflags(new_eflags, FL_UPDATE_CPL0_MASK | VM_MASK | VIF_MASK | VIP_MASK);

    load_seg_vm(R_CS, new_cs);

    cpu_x86_set_cpl(env, 3);

    load_seg_vm(R_SS, new_ss);

    load_seg_vm(R_ES, new_es);

    load_seg_vm(R_DS, new_ds);

    load_seg_vm(R_FS, new_fs);

    load_seg_vm(R_GS, new_gs);



    env->eip = new_eip;

    ESP = new_esp;

}
