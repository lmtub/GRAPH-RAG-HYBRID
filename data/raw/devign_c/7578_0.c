void load_seg(int seg_reg, int selector, unsigned int cur_eip)

{

    uint32_t e1, e2;

    int cpl, dpl, rpl;

    SegmentCache *dt;

    int index;

    uint8_t *ptr;



    if ((selector & 0xfffc) == 0) {

        /* null selector case */

        if (seg_reg == R_SS) {

            EIP = cur_eip;

            raise_exception_err(EXCP0D_GPF, 0);

        } else {

            cpu_x86_load_seg_cache(env, seg_reg, selector, NULL, 0, 0);

        }

    } else {

        

        if (selector & 0x4)

            dt = &env->ldt;

        else

            dt = &env->gdt;

        index = selector & ~7;

        if ((index + 7) > dt->limit) {

            EIP = cur_eip;

            raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        }

        ptr = dt->base + index;

        e1 = ldl_kernel(ptr);

        e2 = ldl_kernel(ptr + 4);



        if (!(e2 & DESC_S_MASK)) {

            EIP = cur_eip;

            raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        }

        rpl = selector & 3;

        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        cpl = env->hflags & HF_CPL_MASK;

        if (seg_reg == R_SS) {

            /* must be writable segment */

            if ((e2 & DESC_CS_MASK) || !(e2 & DESC_W_MASK)) {

                EIP = cur_eip;

                raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

            }

            if (rpl != cpl || dpl != cpl) {

                EIP = cur_eip;

                raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

            }

        } else {

            /* must be readable segment */

            if ((e2 & (DESC_CS_MASK | DESC_R_MASK)) == DESC_CS_MASK) {

                EIP = cur_eip;

                raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

            }

            

            if (!(e2 & DESC_CS_MASK) || !(e2 & DESC_C_MASK)) {

                /* if not conforming code, test rights */

                if (dpl < cpl || dpl < rpl) {

                    EIP = cur_eip;

                    raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

                }

            }

        }



        if (!(e2 & DESC_P_MASK)) {

            EIP = cur_eip;

            if (seg_reg == R_SS)

                raise_exception_err(EXCP0C_STACK, selector & 0xfffc);

            else

                raise_exception_err(EXCP0B_NOSEG, selector & 0xfffc);

        }



        /* set the access bit if not already set */

        if (!(e2 & DESC_A_MASK)) {

            e2 |= DESC_A_MASK;

            stl_kernel(ptr + 4, e2);

        }



        cpu_x86_load_seg_cache(env, seg_reg, selector, 

                       get_seg_base(e1, e2),

                       get_seg_limit(e1, e2),

                       e2);

#if 0

        fprintf(logfile, "load_seg: sel=0x%04x base=0x%08lx limit=0x%08lx flags=%08x\n", 

                selector, (unsigned long)sc->base, sc->limit, sc->flags);

#endif

    }

}
