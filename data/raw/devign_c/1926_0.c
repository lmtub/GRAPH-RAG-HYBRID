void helper_lcall_protected_T0_T1(int shift, int next_eip)

{

    int new_cs, new_eip;

    uint32_t e1, e2, cpl, dpl, rpl, selector, offset, param_count;

    uint32_t ss, ss_e1, ss_e2, push_size, sp, type, ss_dpl;

    uint32_t old_ss, old_esp, val, i, limit;

    uint8_t *ssp, *old_ssp;

    

    new_cs = T0;

    new_eip = T1;

    if ((new_cs & 0xfffc) == 0)

        raise_exception_err(EXCP0D_GPF, 0);

    if (load_segment(&e1, &e2, new_cs) != 0)

        raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

    cpl = env->hflags & HF_CPL_MASK;

    if (e2 & DESC_S_MASK) {

        if (!(e2 & DESC_CS_MASK))

            raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        if (e2 & DESC_CS_MASK) {

            /* conforming code segment */

            if (dpl > cpl)

                raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        } else {

            /* non conforming code segment */

            rpl = new_cs & 3;

            if (rpl > cpl)

                raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

            if (dpl != cpl)

                raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        }

        if (!(e2 & DESC_P_MASK))

            raise_exception_err(EXCP0B_NOSEG, new_cs & 0xfffc);



        sp = ESP;

        if (!(env->segs[R_SS].flags & DESC_B_MASK))

            sp &= 0xffff;

        ssp = env->segs[R_SS].base + sp;

        if (shift) {

            ssp -= 4;

            stl_kernel(ssp, env->segs[R_CS].selector);

            ssp -= 4;

            stl_kernel(ssp, next_eip);

        } else {

            ssp -= 2;

            stw_kernel(ssp, env->segs[R_CS].selector);

            ssp -= 2;

            stw_kernel(ssp, next_eip);

        }

        sp -= (4 << shift);

        

        limit = get_seg_limit(e1, e2);

        if (new_eip > limit)

            raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        /* from this point, not restartable */

        if (!(env->segs[R_SS].flags & DESC_B_MASK))

            ESP = (ESP & 0xffff0000) | (sp & 0xffff);

        else

            ESP = sp;

        cpu_x86_load_seg_cache(env, R_CS, (new_cs & 0xfffc) | cpl,

                       get_seg_base(e1, e2), limit, e2);

        EIP = new_eip;

    } else {

        /* check gate type */

        type = (e2 >> DESC_TYPE_SHIFT) & 0x1f;

        switch(type) {

        case 1: /* available 286 TSS */

        case 9: /* available 386 TSS */

        case 5: /* task gate */

            cpu_abort(env, "task gate not supported");

            break;

        case 4: /* 286 call gate */

        case 12: /* 386 call gate */

            break;

        default:

            raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

            break;

        }

        shift = type >> 3;



        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        rpl = new_cs & 3;

        if (dpl < cpl || dpl < rpl)

            raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        /* check valid bit */

        if (!(e2 & DESC_P_MASK))

            raise_exception_err(EXCP0B_NOSEG,  new_cs & 0xfffc);

        selector = e1 >> 16;

        offset = (e2 & 0xffff0000) | (e1 & 0x0000ffff);

        if ((selector & 0xfffc) == 0)

            raise_exception_err(EXCP0D_GPF, 0);



        if (load_segment(&e1, &e2, selector) != 0)

            raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        if (!(e2 & DESC_S_MASK) || !(e2 & (DESC_CS_MASK)))

            raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        if (dpl > cpl)

            raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        if (!(e2 & DESC_P_MASK))

            raise_exception_err(EXCP0B_NOSEG, selector & 0xfffc);



        if (!(e2 & DESC_C_MASK) && dpl < cpl) {

            /* to inner priviledge */

            get_ss_esp_from_tss(&ss, &sp, dpl);

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

            

            param_count = e2 & 0x1f;

            push_size = ((param_count * 2) + 8) << shift;



            old_esp = ESP;

            old_ss = env->segs[R_SS].selector;

            if (!(env->segs[R_SS].flags & DESC_B_MASK))

                old_esp &= 0xffff;

            old_ssp = env->segs[R_SS].base + old_esp;

            

            /* XXX: from this point not restartable */

            ss = (ss & ~3) | dpl;

            cpu_x86_load_seg_cache(env, R_SS, ss, 

                           get_seg_base(ss_e1, ss_e2),

                           get_seg_limit(ss_e1, ss_e2),

                           ss_e2);



            if (!(env->segs[R_SS].flags & DESC_B_MASK))

                sp &= 0xffff;

            ssp = env->segs[R_SS].base + sp;

            if (shift) {

                ssp -= 4;

                stl_kernel(ssp, old_ss);

                ssp -= 4;

                stl_kernel(ssp, old_esp);

                ssp -= 4 * param_count;

                for(i = 0; i < param_count; i++) {

                    val = ldl_kernel(old_ssp + i * 4);

                    stl_kernel(ssp + i * 4, val);

                }

            } else {

                ssp -= 2;

                stw_kernel(ssp, old_ss);

                ssp -= 2;

                stw_kernel(ssp, old_esp);

                ssp -= 2 * param_count;

                for(i = 0; i < param_count; i++) {

                    val = lduw_kernel(old_ssp + i * 2);

                    stw_kernel(ssp + i * 2, val);

                }

            }

        } else {

            /* to same priviledge */

            if (!(env->segs[R_SS].flags & DESC_B_MASK))

                sp &= 0xffff;

            ssp = env->segs[R_SS].base + sp;

            push_size = (4 << shift);

        }



        if (shift) {

            ssp -= 4;

            stl_kernel(ssp, env->segs[R_CS].selector);

            ssp -= 4;

            stl_kernel(ssp, next_eip);

        } else {

            ssp -= 2;

            stw_kernel(ssp, env->segs[R_CS].selector);

            ssp -= 2;

            stw_kernel(ssp, next_eip);

        }



        sp -= push_size;

        selector = (selector & ~3) | dpl;

        cpu_x86_load_seg_cache(env, R_CS, selector, 

                       get_seg_base(e1, e2),

                       get_seg_limit(e1, e2),

                       e2);

        cpu_x86_set_cpl(env, dpl);

        

        /* from this point, not restartable if same priviledge */

        if (!(env->segs[R_SS].flags & DESC_B_MASK))

            ESP = (ESP & 0xffff0000) | (sp & 0xffff);

        else

            ESP = sp;

        EIP = offset;

    }

}
