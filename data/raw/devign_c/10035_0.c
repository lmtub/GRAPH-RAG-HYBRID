static void do_interrupt_protected(int intno, int is_int, int error_code,

                                   unsigned int next_eip)

{

    SegmentCache *dt;

    uint8_t *ptr, *ssp;

    int type, dpl, selector, ss_dpl;

    int has_error_code, new_stack, shift;

    uint32_t e1, e2, offset, ss, esp, ss_e1, ss_e2, push_size;

    uint32_t old_cs, old_ss, old_esp, old_eip;



    dt = &env->idt;

    if (intno * 8 + 7 > dt->limit)

        raise_exception_err(EXCP0D_GPF, intno * 8 + 2);

    ptr = dt->base + intno * 8;

    e1 = ldl(ptr);

    e2 = ldl(ptr + 4);

    /* check gate type */

    type = (e2 >> DESC_TYPE_SHIFT) & 0x1f;

    switch(type) {

    case 5: /* task gate */

        cpu_abort(env, "task gate not supported");

        break;

    case 6: /* 286 interrupt gate */

    case 7: /* 286 trap gate */

    case 14: /* 386 interrupt gate */

    case 15: /* 386 trap gate */

        break;

    default:

        raise_exception_err(EXCP0D_GPF, intno * 8 + 2);

        break;

    }

    dpl = (e2 >> DESC_DPL_SHIFT) & 3;

    /* check privledge if software int */

    if (is_int && dpl < env->cpl)

        raise_exception_err(EXCP0D_GPF, intno * 8 + 2);

    /* check valid bit */

    if (!(e2 & DESC_P_MASK))

        raise_exception_err(EXCP0B_NOSEG, intno * 8 + 2);

    selector = e1 >> 16;

    offset = (e2 & 0xffff0000) | (e1 & 0x0000ffff);

    if ((selector & 0xfffc) == 0)

        raise_exception_err(EXCP0D_GPF, 0);



    if (load_segment(&e1, &e2, selector) != 0)

        raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

    if (!(e2 & DESC_S_MASK) || !(e2 & (DESC_CS_MASK)))

        raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

    dpl = (e2 >> DESC_DPL_SHIFT) & 3;

    if (dpl > env->cpl)

        raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

    if (!(e2 & DESC_P_MASK))

        raise_exception_err(EXCP0B_NOSEG, selector & 0xfffc);

    if (!(e2 & DESC_C_MASK) && dpl < env->cpl) {

        /* to inner priviledge */

        get_ss_esp_from_tss(&ss, &esp, dpl);

        if ((ss & 0xfffc) == 0)

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        if ((ss & 3) != dpl)

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        if (load_segment(&ss_e1, &ss_e2, ss) != 0)

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        ss_dpl = (ss_e2 >> DESC_DPL_SHIFT) & 3;

        if (ss_dpl != dpl)

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        if (!(ss_e2 & DESC_S_MASK) ||

            (ss_e2 & DESC_CS_MASK) ||

            !(ss_e2 & DESC_W_MASK))

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        if (!(ss_e2 & DESC_P_MASK))

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        new_stack = 1;

    } else if ((e2 & DESC_C_MASK) || dpl == env->cpl) {

        /* to same priviledge */

        new_stack = 0;

    } else {

        raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        new_stack = 0; /* avoid warning */

    }



    shift = type >> 3;

    has_error_code = 0;

    if (!is_int) {

        switch(intno) {

        case 8:

        case 10:

        case 11:

        case 12:

        case 13:

        case 14:

        case 17:

            has_error_code = 1;

            break;

        }

    }

    push_size = 6 + (new_stack << 2) + (has_error_code << 1);

    if (env->eflags & VM_MASK)

        push_size += 8;

    push_size <<= shift;



    /* XXX: check that enough room is available */

    if (new_stack) {

        old_esp = env->regs[R_ESP];

        old_ss = env->segs[R_SS].selector;

        load_seg(R_SS, ss, env->eip);

    } else {

        old_esp = 0;

        old_ss = 0;

        esp = env->regs[R_ESP];

    }

    if (is_int)

        old_eip = next_eip;

    else

        old_eip = env->eip;

    old_cs = env->segs[R_CS].selector;

    load_seg(R_CS, selector, env->eip);

    env->eip = offset;

    env->regs[R_ESP] = esp - push_size;

    ssp = env->segs[R_SS].base + esp;

    if (shift == 1) {

        int old_eflags;

        if (env->eflags & VM_MASK) {

            ssp -= 4;

            stl(ssp, env->segs[R_GS].selector);

            ssp -= 4;

            stl(ssp, env->segs[R_FS].selector);

            ssp -= 4;

            stl(ssp, env->segs[R_DS].selector);

            ssp -= 4;

            stl(ssp, env->segs[R_ES].selector);

        }

        if (new_stack) {

            ssp -= 4;

            stl(ssp, old_ss);

            ssp -= 4;

            stl(ssp, old_esp);

        }

        ssp -= 4;

        old_eflags = compute_eflags();

        stl(ssp, old_eflags);

        ssp -= 4;

        stl(ssp, old_cs);

        ssp -= 4;

        stl(ssp, old_eip);

        if (has_error_code) {

            ssp -= 4;

            stl(ssp, error_code);

        }

    } else {

        if (new_stack) {

            ssp -= 2;

            stw(ssp, old_ss);

            ssp -= 2;

            stw(ssp, old_esp);

        }

        ssp -= 2;

        stw(ssp, compute_eflags());

        ssp -= 2;

        stw(ssp, old_cs);

        ssp -= 2;

        stw(ssp, old_eip);

        if (has_error_code) {

            ssp -= 2;

            stw(ssp, error_code);

        }

    }

    

    /* interrupt gate clear IF mask */

    if ((type & 1) == 0) {

        env->eflags &= ~IF_MASK;

    }

    env->eflags &= ~(TF_MASK | VM_MASK | RF_MASK | NT_MASK);

}
