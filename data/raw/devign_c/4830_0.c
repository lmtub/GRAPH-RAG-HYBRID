static void switch_tss(int tss_selector, 

                       uint32_t e1, uint32_t e2, int source)

{

    int tss_limit, tss_limit_max, type, old_tss_limit_max, old_type, v1, v2, i;

    uint8_t *tss_base;

    uint32_t new_regs[8], new_segs[6];

    uint32_t new_eflags, new_eip, new_cr3, new_ldt, new_trap;

    uint32_t old_eflags, eflags_mask;

    SegmentCache *dt;

    int index;

    uint8_t *ptr;



    type = (e2 >> DESC_TYPE_SHIFT) & 0xf;

#ifdef DEBUG_PCALL

    if (loglevel)

        fprintf(logfile, "switch_tss: sel=0x%04x type=%d src=%d\n", tss_selector, type, source);

#endif



    /* if task gate, we read the TSS segment and we load it */

    if (type == 5) {

        if (!(e2 & DESC_P_MASK))

            raise_exception_err(EXCP0B_NOSEG, tss_selector & 0xfffc);

        tss_selector = e1 >> 16;

        if (tss_selector & 4)

            raise_exception_err(EXCP0A_TSS, tss_selector & 0xfffc);

        if (load_segment(&e1, &e2, tss_selector) != 0)

            raise_exception_err(EXCP0D_GPF, tss_selector & 0xfffc);

        if (e2 & DESC_S_MASK)

            raise_exception_err(EXCP0D_GPF, tss_selector & 0xfffc);

        type = (e2 >> DESC_TYPE_SHIFT) & 0xf;

        if ((type & 7) != 1)

            raise_exception_err(EXCP0D_GPF, tss_selector & 0xfffc);

    }



    if (!(e2 & DESC_P_MASK))

        raise_exception_err(EXCP0B_NOSEG, tss_selector & 0xfffc);



    if (type & 8)

        tss_limit_max = 103;

    else

        tss_limit_max = 43;

    tss_limit = get_seg_limit(e1, e2);

    tss_base = get_seg_base(e1, e2);

    if ((tss_selector & 4) != 0 || 

        tss_limit < tss_limit_max)

        raise_exception_err(EXCP0A_TSS, tss_selector & 0xfffc);

    old_type = (env->tr.flags >> DESC_TYPE_SHIFT) & 0xf;

    if (old_type & 8)

        old_tss_limit_max = 103;

    else

        old_tss_limit_max = 43;



    /* read all the registers from the new TSS */

    if (type & 8) {

        /* 32 bit */

        new_cr3 = ldl_kernel(tss_base + 0x1c);

        new_eip = ldl_kernel(tss_base + 0x20);

        new_eflags = ldl_kernel(tss_base + 0x24);

        for(i = 0; i < 8; i++)

            new_regs[i] = ldl_kernel(tss_base + (0x28 + i * 4));

        for(i = 0; i < 6; i++)

            new_segs[i] = lduw_kernel(tss_base + (0x48 + i * 4));

        new_ldt = lduw_kernel(tss_base + 0x60);

        new_trap = ldl_kernel(tss_base + 0x64);

    } else {

        /* 16 bit */

        new_cr3 = 0;

        new_eip = lduw_kernel(tss_base + 0x0e);

        new_eflags = lduw_kernel(tss_base + 0x10);

        for(i = 0; i < 8; i++)

            new_regs[i] = lduw_kernel(tss_base + (0x12 + i * 2)) | 0xffff0000;

        for(i = 0; i < 4; i++)

            new_segs[i] = lduw_kernel(tss_base + (0x22 + i * 4));

        new_ldt = lduw_kernel(tss_base + 0x2a);

        new_segs[R_FS] = 0;

        new_segs[R_GS] = 0;

        new_trap = 0;

    }

    

    /* NOTE: we must avoid memory exceptions during the task switch,

       so we make dummy accesses before */

    /* XXX: it can still fail in some cases, so a bigger hack is

       necessary to valid the TLB after having done the accesses */



    v1 = ldub_kernel(env->tr.base);

    v2 = ldub(env->tr.base + old_tss_limit_max);

    stb_kernel(env->tr.base, v1);

    stb_kernel(env->tr.base + old_tss_limit_max, v2);

    

    /* clear busy bit (it is restartable) */

    if (source == SWITCH_TSS_JMP || source == SWITCH_TSS_IRET) {

        uint8_t *ptr;

        uint32_t e2;

        ptr = env->gdt.base + (env->tr.selector << 3);

        e2 = ldl_kernel(ptr + 4);

        e2 &= ~DESC_TSS_BUSY_MASK;

        stl_kernel(ptr + 4, e2);

    }

    old_eflags = compute_eflags();

    if (source == SWITCH_TSS_IRET)

        old_eflags &= ~NT_MASK;

    

    /* save the current state in the old TSS */

    if (type & 8) {

        /* 32 bit */

        stl_kernel(env->tr.base + 0x20, env->eip);

        stl_kernel(env->tr.base + 0x24, old_eflags);

        for(i = 0; i < 8; i++)

            stl_kernel(env->tr.base + (0x28 + i * 4), env->regs[i]);

        for(i = 0; i < 6; i++)

            stw_kernel(env->tr.base + (0x48 + i * 4), env->segs[i].selector);

    } else {

        /* 16 bit */

        stw_kernel(env->tr.base + 0x0e, new_eip);

        stw_kernel(env->tr.base + 0x10, old_eflags);

        for(i = 0; i < 8; i++)

            stw_kernel(env->tr.base + (0x12 + i * 2), env->regs[i]);

        for(i = 0; i < 4; i++)

            stw_kernel(env->tr.base + (0x22 + i * 4), env->segs[i].selector);

    }

    

    /* now if an exception occurs, it will occurs in the next task

       context */



    if (source == SWITCH_TSS_CALL) {

        stw_kernel(tss_base, env->tr.selector);

        new_eflags |= NT_MASK;

    }



    /* set busy bit */

    if (source == SWITCH_TSS_JMP || source == SWITCH_TSS_CALL) {

        uint8_t *ptr;

        uint32_t e2;

        ptr = env->gdt.base + (tss_selector << 3);

        e2 = ldl_kernel(ptr + 4);

        e2 |= DESC_TSS_BUSY_MASK;

        stl_kernel(ptr + 4, e2);

    }



    /* set the new CPU state */

    /* from this point, any exception which occurs can give problems */

    env->cr[0] |= CR0_TS_MASK;

    env->tr.selector = tss_selector;

    env->tr.base = tss_base;

    env->tr.limit = tss_limit;

    env->tr.flags = e2 & ~DESC_TSS_BUSY_MASK;

    

    if ((type & 8) && (env->cr[0] & CR0_PG_MASK)) {

        env->cr[3] = new_cr3;

        cpu_x86_update_cr3(env);

    }

    

    /* load all registers without an exception, then reload them with

       possible exception */

    env->eip = new_eip;

    eflags_mask = TF_MASK | AC_MASK | ID_MASK | 

        IF_MASK | IOPL_MASK | VM_MASK | RF_MASK | NT_MASK;

    if (!(type & 8))

        eflags_mask &= 0xffff;

    load_eflags(new_eflags, eflags_mask);

    for(i = 0; i < 8; i++)

        env->regs[i] = new_regs[i];

    if (new_eflags & VM_MASK) {

        for(i = 0; i < 6; i++) 

            load_seg_vm(i, new_segs[i]);

        /* in vm86, CPL is always 3 */

        cpu_x86_set_cpl(env, 3);

    } else {

        /* CPL is set the RPL of CS */

        cpu_x86_set_cpl(env, new_segs[R_CS] & 3);

        /* first just selectors as the rest may trigger exceptions */

        for(i = 0; i < 6; i++)

            cpu_x86_load_seg_cache(env, i, new_segs[i], NULL, 0, 0);

    }

    

    env->ldt.selector = new_ldt & ~4;

    env->ldt.base = NULL;

    env->ldt.limit = 0;

    env->ldt.flags = 0;



    /* load the LDT */

    if (new_ldt & 4)

        raise_exception_err(EXCP0A_TSS, new_ldt & 0xfffc);



    if ((new_ldt & 0xfffc) != 0) {

        dt = &env->gdt;

        index = new_ldt & ~7;

        if ((index + 7) > dt->limit)

            raise_exception_err(EXCP0A_TSS, new_ldt & 0xfffc);

        ptr = dt->base + index;

        e1 = ldl_kernel(ptr);

        e2 = ldl_kernel(ptr + 4);

        if ((e2 & DESC_S_MASK) || ((e2 >> DESC_TYPE_SHIFT) & 0xf) != 2)

            raise_exception_err(EXCP0A_TSS, new_ldt & 0xfffc);

        if (!(e2 & DESC_P_MASK))

            raise_exception_err(EXCP0A_TSS, new_ldt & 0xfffc);

        load_seg_cache_raw_dt(&env->ldt, e1, e2);

    }

    

    /* load the segments */

    if (!(new_eflags & VM_MASK)) {

        tss_load_seg(R_CS, new_segs[R_CS]);

        tss_load_seg(R_SS, new_segs[R_SS]);

        tss_load_seg(R_ES, new_segs[R_ES]);

        tss_load_seg(R_DS, new_segs[R_DS]);

        tss_load_seg(R_FS, new_segs[R_FS]);

        tss_load_seg(R_GS, new_segs[R_GS]);

    }

    

    /* check that EIP is in the CS segment limits */

    if (new_eip > env->segs[R_CS].limit) {

        raise_exception_err(EXCP0D_GPF, 0);

    }

}
