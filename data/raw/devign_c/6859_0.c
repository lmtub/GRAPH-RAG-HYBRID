void helper_lcall_protected(CPUX86State *env, int new_cs, target_ulong new_eip,

                            int shift, int next_eip_addend)

{

    int new_stack, i;

    uint32_t e1, e2, cpl, dpl, rpl, selector, offset, param_count;

    uint32_t ss = 0, ss_e1 = 0, ss_e2 = 0, sp, type, ss_dpl, sp_mask;

    uint32_t val, limit, old_sp_mask;

    target_ulong ssp, old_ssp, next_eip;



    next_eip = env->eip + next_eip_addend;

    LOG_PCALL("lcall %04x:%08x s=%d\n", new_cs, (uint32_t)new_eip, shift);

    LOG_PCALL_STATE(CPU(x86_env_get_cpu(env)));

    if ((new_cs & 0xfffc) == 0) {

        raise_exception_err(env, EXCP0D_GPF, 0);

    }

    if (load_segment(env, &e1, &e2, new_cs) != 0) {

        raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

    }

    cpl = env->hflags & HF_CPL_MASK;

    LOG_PCALL("desc=%08x:%08x\n", e1, e2);

    if (e2 & DESC_S_MASK) {

        if (!(e2 & DESC_CS_MASK)) {

            raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

        }

        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        if (e2 & DESC_C_MASK) {

            /* conforming code segment */

            if (dpl > cpl) {

                raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

            }

        } else {

            /* non conforming code segment */

            rpl = new_cs & 3;

            if (rpl > cpl) {

                raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

            }

            if (dpl != cpl) {

                raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

            }

        }

        if (!(e2 & DESC_P_MASK)) {

            raise_exception_err(env, EXCP0B_NOSEG, new_cs & 0xfffc);

        }



#ifdef TARGET_X86_64

        /* XXX: check 16/32 bit cases in long mode */

        if (shift == 2) {

            target_ulong rsp;



            /* 64 bit case */

            rsp = env->regs[R_ESP];

            PUSHQ(rsp, env->segs[R_CS].selector);

            PUSHQ(rsp, next_eip);

            /* from this point, not restartable */

            env->regs[R_ESP] = rsp;

            cpu_x86_load_seg_cache(env, R_CS, (new_cs & 0xfffc) | cpl,

                                   get_seg_base(e1, e2),

                                   get_seg_limit(e1, e2), e2);

            env->eip = new_eip;

        } else

#endif

        {

            sp = env->regs[R_ESP];

            sp_mask = get_sp_mask(env->segs[R_SS].flags);

            ssp = env->segs[R_SS].base;

            if (shift) {

                PUSHL(ssp, sp, sp_mask, env->segs[R_CS].selector);

                PUSHL(ssp, sp, sp_mask, next_eip);

            } else {

                PUSHW(ssp, sp, sp_mask, env->segs[R_CS].selector);

                PUSHW(ssp, sp, sp_mask, next_eip);

            }



            limit = get_seg_limit(e1, e2);

            if (new_eip > limit) {

                raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

            }

            /* from this point, not restartable */

            SET_ESP(sp, sp_mask);

            cpu_x86_load_seg_cache(env, R_CS, (new_cs & 0xfffc) | cpl,

                                   get_seg_base(e1, e2), limit, e2);

            env->eip = new_eip;

        }

    } else {

        /* check gate type */

        type = (e2 >> DESC_TYPE_SHIFT) & 0x1f;

        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        rpl = new_cs & 3;

        switch (type) {

        case 1: /* available 286 TSS */

        case 9: /* available 386 TSS */

        case 5: /* task gate */

            if (dpl < cpl || dpl < rpl) {

                raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

            }

            switch_tss(env, new_cs, e1, e2, SWITCH_TSS_CALL, next_eip);

            CC_OP = CC_OP_EFLAGS;

            return;

        case 4: /* 286 call gate */

        case 12: /* 386 call gate */

            break;

        default:

            raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

            break;

        }

        shift = type >> 3;



        if (dpl < cpl || dpl < rpl) {

            raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

        }

        /* check valid bit */

        if (!(e2 & DESC_P_MASK)) {

            raise_exception_err(env, EXCP0B_NOSEG,  new_cs & 0xfffc);

        }

        selector = e1 >> 16;

        offset = (e2 & 0xffff0000) | (e1 & 0x0000ffff);

        param_count = e2 & 0x1f;

        if ((selector & 0xfffc) == 0) {

            raise_exception_err(env, EXCP0D_GPF, 0);

        }



        if (load_segment(env, &e1, &e2, selector) != 0) {

            raise_exception_err(env, EXCP0D_GPF, selector & 0xfffc);

        }

        if (!(e2 & DESC_S_MASK) || !(e2 & (DESC_CS_MASK))) {

            raise_exception_err(env, EXCP0D_GPF, selector & 0xfffc);

        }

        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        if (dpl > cpl) {

            raise_exception_err(env, EXCP0D_GPF, selector & 0xfffc);

        }

        if (!(e2 & DESC_P_MASK)) {

            raise_exception_err(env, EXCP0B_NOSEG, selector & 0xfffc);

        }



        if (!(e2 & DESC_C_MASK) && dpl < cpl) {

            /* to inner privilege */

            get_ss_esp_from_tss(env, &ss, &sp, dpl);

            LOG_PCALL("new ss:esp=%04x:%08x param_count=%d env->regs[R_ESP]="

                      TARGET_FMT_lx "\n", ss, sp, param_count,

                      env->regs[R_ESP]);

            if ((ss & 0xfffc) == 0) {

                raise_exception_err(env, EXCP0A_TSS, ss & 0xfffc);

            }

            if ((ss & 3) != dpl) {

                raise_exception_err(env, EXCP0A_TSS, ss & 0xfffc);

            }

            if (load_segment(env, &ss_e1, &ss_e2, ss) != 0) {

                raise_exception_err(env, EXCP0A_TSS, ss & 0xfffc);

            }

            ss_dpl = (ss_e2 >> DESC_DPL_SHIFT) & 3;

            if (ss_dpl != dpl) {

                raise_exception_err(env, EXCP0A_TSS, ss & 0xfffc);

            }

            if (!(ss_e2 & DESC_S_MASK) ||

                (ss_e2 & DESC_CS_MASK) ||

                !(ss_e2 & DESC_W_MASK)) {

                raise_exception_err(env, EXCP0A_TSS, ss & 0xfffc);

            }

            if (!(ss_e2 & DESC_P_MASK)) {

                raise_exception_err(env, EXCP0A_TSS, ss & 0xfffc);

            }



            /* push_size = ((param_count * 2) + 8) << shift; */



            old_sp_mask = get_sp_mask(env->segs[R_SS].flags);

            old_ssp = env->segs[R_SS].base;



            sp_mask = get_sp_mask(ss_e2);

            ssp = get_seg_base(ss_e1, ss_e2);

            if (shift) {

                PUSHL(ssp, sp, sp_mask, env->segs[R_SS].selector);

                PUSHL(ssp, sp, sp_mask, env->regs[R_ESP]);

                for (i = param_count - 1; i >= 0; i--) {

                    val = cpu_ldl_kernel(env, old_ssp +

                                         ((env->regs[R_ESP] + i * 4) &

                                          old_sp_mask));

                    PUSHL(ssp, sp, sp_mask, val);

                }

            } else {

                PUSHW(ssp, sp, sp_mask, env->segs[R_SS].selector);

                PUSHW(ssp, sp, sp_mask, env->regs[R_ESP]);

                for (i = param_count - 1; i >= 0; i--) {

                    val = cpu_lduw_kernel(env, old_ssp +

                                          ((env->regs[R_ESP] + i * 2) &

                                           old_sp_mask));

                    PUSHW(ssp, sp, sp_mask, val);

                }

            }

            new_stack = 1;

        } else {

            /* to same privilege */

            sp = env->regs[R_ESP];

            sp_mask = get_sp_mask(env->segs[R_SS].flags);

            ssp = env->segs[R_SS].base;

            /* push_size = (4 << shift); */

            new_stack = 0;

        }



        if (shift) {

            PUSHL(ssp, sp, sp_mask, env->segs[R_CS].selector);

            PUSHL(ssp, sp, sp_mask, next_eip);

        } else {

            PUSHW(ssp, sp, sp_mask, env->segs[R_CS].selector);

            PUSHW(ssp, sp, sp_mask, next_eip);

        }



        /* from this point, not restartable */



        if (new_stack) {

            ss = (ss & ~3) | dpl;

            cpu_x86_load_seg_cache(env, R_SS, ss,

                                   ssp,

                                   get_seg_limit(ss_e1, ss_e2),

                                   ss_e2);

        }



        selector = (selector & ~3) | dpl;

        cpu_x86_load_seg_cache(env, R_CS, selector,

                       get_seg_base(e1, e2),

                       get_seg_limit(e1, e2),

                       e2);

        cpu_x86_set_cpl(env, dpl);

        SET_ESP(sp, sp_mask);

        env->eip = offset;

    }

}
