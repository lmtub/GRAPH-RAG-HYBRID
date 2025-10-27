static int cpu_gdb_write_register(CPUState *env, uint8_t *mem_buf, int n)

{

    uint32_t tmp;



    if (n < CPU_NB_REGS) {

        env->regs[gpr_map[n]] = ldtul_p(mem_buf);

        return sizeof(target_ulong);

    } else if (n >= IDX_FP_REGS && n < IDX_FP_REGS + 8) {

#ifdef USE_X86LDOUBLE

        /* FIXME: byteswap float values - after fixing fpregs layout. */

        memcpy(&env->fpregs[n - IDX_FP_REGS], mem_buf, 10);

#endif

        return 10;

    } else if (n >= IDX_XMM_REGS && n < IDX_XMM_REGS + CPU_NB_REGS) {

        n -= IDX_XMM_REGS;

        env->xmm_regs[n].XMM_Q(0) = ldq_p(mem_buf);

        env->xmm_regs[n].XMM_Q(1) = ldq_p(mem_buf + 8);

        return 16;

    } else {

        switch (n) {

        case IDX_IP_REG:

            env->eip = ldtul_p(mem_buf);

            return sizeof(target_ulong);

        case IDX_FLAGS_REG:

            env->eflags = ldl_p(mem_buf);

            return 4;



#if defined(CONFIG_USER_ONLY)

#define LOAD_SEG(index, sreg)\

            tmp = ldl_p(mem_buf);\

            if (tmp != env->segs[sreg].selector)\

                cpu_x86_load_seg(env, sreg, tmp);\

            return 4

#else

/* FIXME: Honor segment registers.  Needs to avoid raising an exception

   when the selector is invalid.  */

#define LOAD_SEG(index, sreg) return 4

#endif

        case IDX_SEG_REGS:     LOAD_SEG(10, R_CS);

        case IDX_SEG_REGS + 1: LOAD_SEG(11, R_SS);

        case IDX_SEG_REGS + 2: LOAD_SEG(12, R_DS);

        case IDX_SEG_REGS + 3: LOAD_SEG(13, R_ES);

        case IDX_SEG_REGS + 4: LOAD_SEG(14, R_FS);

        case IDX_SEG_REGS + 5: LOAD_SEG(15, R_GS);



        case IDX_FP_REGS + 8:

            env->fpuc = ldl_p(mem_buf);

            return 4;

        case IDX_FP_REGS + 9:

            tmp = ldl_p(mem_buf);

            env->fpstt = (tmp >> 11) & 7;

            env->fpus = tmp & ~0x3800;

            return 4;

        case IDX_FP_REGS + 10: /* ftag */  return 4;

        case IDX_FP_REGS + 11: /* fiseg */ return 4;

        case IDX_FP_REGS + 12: /* fioff */ return 4;

        case IDX_FP_REGS + 13: /* foseg */ return 4;

        case IDX_FP_REGS + 14: /* fooff */ return 4;

        case IDX_FP_REGS + 15: /* fop */   return 4;



        case IDX_MXCSR_REG:

            env->mxcsr = ldl_p(mem_buf);

            return 4;

        }

    }

    /* Unrecognised register.  */

    return 0;

}
