void do_interrupt (CPUState *env)

{

#if !defined(CONFIG_USER_ONLY)

    target_ulong offset;

    int cause = -1;

    const char *name;



    if (qemu_log_enabled() && env->exception_index != EXCP_EXT_INTERRUPT) {

        if (env->exception_index < 0 || env->exception_index > EXCP_LAST)

            name = "unknown";

        else

            name = excp_names[env->exception_index];



        qemu_log("%s enter: PC " TARGET_FMT_lx " EPC " TARGET_FMT_lx " %s exception\n",

                 __func__, env->active_tc.PC, env->CP0_EPC, name);

    }

    if (env->exception_index == EXCP_EXT_INTERRUPT &&

        (env->hflags & MIPS_HFLAG_DM))

        env->exception_index = EXCP_DINT;

    offset = 0x180;

    switch (env->exception_index) {

    case EXCP_DSS:

        env->CP0_Debug |= 1 << CP0DB_DSS;

        /* Debug single step cannot be raised inside a delay slot and

           resume will always occur on the next instruction

           (but we assume the pc has always been updated during

           code translation). */

        env->CP0_DEPC = env->active_tc.PC | !!(env->hflags & MIPS_HFLAG_M16);

        goto enter_debug_mode;

    case EXCP_DINT:

        env->CP0_Debug |= 1 << CP0DB_DINT;

        goto set_DEPC;

    case EXCP_DIB:

        env->CP0_Debug |= 1 << CP0DB_DIB;

        goto set_DEPC;

    case EXCP_DBp:

        env->CP0_Debug |= 1 << CP0DB_DBp;

        goto set_DEPC;

    case EXCP_DDBS:

        env->CP0_Debug |= 1 << CP0DB_DDBS;

        goto set_DEPC;

    case EXCP_DDBL:

        env->CP0_Debug |= 1 << CP0DB_DDBL;

    set_DEPC:

        env->CP0_DEPC = exception_resume_pc(env);

        env->hflags &= ~MIPS_HFLAG_BMASK;

 enter_debug_mode:

        env->hflags |= MIPS_HFLAG_DM | MIPS_HFLAG_64 | MIPS_HFLAG_CP0;

        env->hflags &= ~(MIPS_HFLAG_KSU);

        /* EJTAG probe trap enable is not implemented... */

        if (!(env->CP0_Status & (1 << CP0St_EXL)))

            env->CP0_Cause &= ~(1 << CP0Ca_BD);

        env->active_tc.PC = (int32_t)0xBFC00480;

        set_hflags_for_handler(env);

        break;

    case EXCP_RESET:

        cpu_reset(env);

        break;

    case EXCP_SRESET:

        env->CP0_Status |= (1 << CP0St_SR);

        memset(env->CP0_WatchLo, 0, sizeof(*env->CP0_WatchLo));

        goto set_error_EPC;

    case EXCP_NMI:

        env->CP0_Status |= (1 << CP0St_NMI);

 set_error_EPC:

        env->CP0_ErrorEPC = exception_resume_pc(env);

        env->hflags &= ~MIPS_HFLAG_BMASK;

        env->CP0_Status |= (1 << CP0St_ERL) | (1 << CP0St_BEV);

        env->hflags |= MIPS_HFLAG_64 | MIPS_HFLAG_CP0;

        env->hflags &= ~(MIPS_HFLAG_KSU);

        if (!(env->CP0_Status & (1 << CP0St_EXL)))

            env->CP0_Cause &= ~(1 << CP0Ca_BD);

        env->active_tc.PC = (int32_t)0xBFC00000;

        set_hflags_for_handler(env);

        break;

    case EXCP_EXT_INTERRUPT:

        cause = 0;

        if (env->CP0_Cause & (1 << CP0Ca_IV))

            offset = 0x200;



        if (env->CP0_Config3 & ((1 << CP0C3_VInt) | (1 << CP0C3_VEIC))) {

            /* Vectored Interrupts.  */

            unsigned int spacing;

            unsigned int vector;

            unsigned int pending = (env->CP0_Cause & CP0Ca_IP_mask) >> 8;



            pending &= env->CP0_Status >> 8;

            /* Compute the Vector Spacing.  */

            spacing = (env->CP0_IntCtl >> CP0IntCtl_VS) & ((1 << 6) - 1);

            spacing <<= 5;



            if (env->CP0_Config3 & (1 << CP0C3_VInt)) {

                /* For VInt mode, the MIPS computes the vector internally.  */

                for (vector = 7; vector > 0; vector--) {

                    if (pending & (1 << vector)) {

                        /* Found it.  */

                        break;

                    }

                }

            } else {

                /* For VEIC mode, the external interrupt controller feeds the

                   vector throught the CP0Cause IP lines.  */

                vector = pending;

            }

            offset = 0x200 + vector * spacing;

        }

        goto set_EPC;

    case EXCP_LTLBL:

        cause = 1;

        goto set_EPC;

    case EXCP_TLBL:

        cause = 2;

        if (env->error_code == 1 && !(env->CP0_Status & (1 << CP0St_EXL))) {

#if defined(TARGET_MIPS64)

            int R = env->CP0_BadVAddr >> 62;

            int UX = (env->CP0_Status & (1 << CP0St_UX)) != 0;

            int SX = (env->CP0_Status & (1 << CP0St_SX)) != 0;

            int KX = (env->CP0_Status & (1 << CP0St_KX)) != 0;



            if (((R == 0 && UX) || (R == 1 && SX) || (R == 3 && KX)) &&

                (!(env->insn_flags & (INSN_LOONGSON2E | INSN_LOONGSON2F))))

                offset = 0x080;

            else

#endif

                offset = 0x000;

        }

        goto set_EPC;

    case EXCP_TLBS:

        cause = 3;

        if (env->error_code == 1 && !(env->CP0_Status & (1 << CP0St_EXL))) {

#if defined(TARGET_MIPS64)

            int R = env->CP0_BadVAddr >> 62;

            int UX = (env->CP0_Status & (1 << CP0St_UX)) != 0;

            int SX = (env->CP0_Status & (1 << CP0St_SX)) != 0;

            int KX = (env->CP0_Status & (1 << CP0St_KX)) != 0;



            if (((R == 0 && UX) || (R == 1 && SX) || (R == 3 && KX)) &&

                (!(env->insn_flags & (INSN_LOONGSON2E | INSN_LOONGSON2F))))

                offset = 0x080;

            else

#endif

                offset = 0x000;

        }

        goto set_EPC;

    case EXCP_AdEL:

        cause = 4;

        goto set_EPC;

    case EXCP_AdES:

        cause = 5;

        goto set_EPC;

    case EXCP_IBE:

        cause = 6;

        goto set_EPC;

    case EXCP_DBE:

        cause = 7;

        goto set_EPC;

    case EXCP_SYSCALL:

        cause = 8;

        goto set_EPC;

    case EXCP_BREAK:

        cause = 9;

        goto set_EPC;

    case EXCP_RI:

        cause = 10;

        goto set_EPC;

    case EXCP_CpU:

        cause = 11;

        env->CP0_Cause = (env->CP0_Cause & ~(0x3 << CP0Ca_CE)) |

                         (env->error_code << CP0Ca_CE);

        goto set_EPC;

    case EXCP_OVERFLOW:

        cause = 12;

        goto set_EPC;

    case EXCP_TRAP:

        cause = 13;

        goto set_EPC;

    case EXCP_FPE:

        cause = 15;

        goto set_EPC;

    case EXCP_C2E:

        cause = 18;

        goto set_EPC;

    case EXCP_MDMX:

        cause = 22;

        goto set_EPC;

    case EXCP_DWATCH:

        cause = 23;

        /* XXX: TODO: manage defered watch exceptions */

        goto set_EPC;

    case EXCP_MCHECK:

        cause = 24;

        goto set_EPC;

    case EXCP_THREAD:

        cause = 25;

        goto set_EPC;

    case EXCP_CACHE:

        cause = 30;

        if (env->CP0_Status & (1 << CP0St_BEV)) {

            offset = 0x100;

        } else {

            offset = 0x20000100;

        }

 set_EPC:

        if (!(env->CP0_Status & (1 << CP0St_EXL))) {

            env->CP0_EPC = exception_resume_pc(env);

            if (env->hflags & MIPS_HFLAG_BMASK) {

                env->CP0_Cause |= (1 << CP0Ca_BD);

            } else {

                env->CP0_Cause &= ~(1 << CP0Ca_BD);

            }

            env->CP0_Status |= (1 << CP0St_EXL);

            env->hflags |= MIPS_HFLAG_64 | MIPS_HFLAG_CP0;

            env->hflags &= ~(MIPS_HFLAG_KSU);

        }

        env->hflags &= ~MIPS_HFLAG_BMASK;

        if (env->CP0_Status & (1 << CP0St_BEV)) {

            env->active_tc.PC = (int32_t)0xBFC00200;

        } else {

            env->active_tc.PC = (int32_t)(env->CP0_EBase & ~0x3ff);

        }

        env->active_tc.PC += offset;

        set_hflags_for_handler(env);

        env->CP0_Cause = (env->CP0_Cause & ~(0x1f << CP0Ca_EC)) | (cause << CP0Ca_EC);

        break;

    default:

        qemu_log("Invalid MIPS exception %d. Exiting\n", env->exception_index);

        printf("Invalid MIPS exception %d. Exiting\n", env->exception_index);

        exit(1);

    }

    if (qemu_log_enabled() && env->exception_index != EXCP_EXT_INTERRUPT) {

        qemu_log("%s: PC " TARGET_FMT_lx " EPC " TARGET_FMT_lx " cause %d\n"

                "    S %08x C %08x A " TARGET_FMT_lx " D " TARGET_FMT_lx "\n",

                __func__, env->active_tc.PC, env->CP0_EPC, cause,

                env->CP0_Status, env->CP0_Cause, env->CP0_BadVAddr,

                env->CP0_DEPC);

    }

#endif

    env->exception_index = EXCP_NONE;

}
