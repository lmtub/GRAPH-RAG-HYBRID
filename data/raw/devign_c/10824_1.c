void do_interrupt(CPUState *env)

{

    int intno = env->exception_index;



#ifdef DEBUG_PCALL

    if (qemu_loglevel_mask(CPU_LOG_INT)) {

        static int count;

        const char *name;



        if (intno < 0 || intno >= 0x180)

            name = "Unknown";

        else if (intno >= 0x100)

            name = "Trap Instruction";

        else if (intno >= 0xc0)

            name = "Window Fill";

        else if (intno >= 0x80)

            name = "Window Spill";

        else {

            name = excp_names[intno];

            if (!name)

                name = "Unknown";

        }



        qemu_log("%6d: %s (v=%04x) pc=%016" PRIx64 " npc=%016" PRIx64

                " SP=%016" PRIx64 "\n",

                count, name, intno,

                env->pc,

                env->npc, env->regwptr[6]);

        log_cpu_state(env, 0);

#if 0

        {

            int i;

            uint8_t *ptr;



            qemu_log("       code=");

            ptr = (uint8_t *)env->pc;

            for(i = 0; i < 16; i++) {

                qemu_log(" %02x", ldub(ptr + i));

            }

            qemu_log("\n");

        }

#endif

        count++;

    }

#endif

#if !defined(CONFIG_USER_ONLY)

    if (env->tl >= env->maxtl) {

        cpu_abort(env, "Trap 0x%04x while trap level (%d) >= MAXTL (%d),"

                  " Error state", env->exception_index, env->tl, env->maxtl);

        return;

    }

#endif

    if (env->tl < env->maxtl - 1) {

        env->tl++;

    } else {

        env->pstate |= PS_RED;

        if (env->tl < env->maxtl)

            env->tl++;

    }

    env->tsptr = &env->ts[env->tl & MAXTL_MASK];

    env->tsptr->tstate = ((uint64_t)GET_CCR(env) << 32) |

        ((env->asi & 0xff) << 24) | ((env->pstate & 0xf3f) << 8) |

        GET_CWP64(env);

    env->tsptr->tpc = env->pc;

    env->tsptr->tnpc = env->npc;

    env->tsptr->tt = intno;

    if (!(env->def->features & CPU_FEATURE_GL)) {

        switch (intno) {

        case TT_IVEC:

            change_pstate(PS_PEF | PS_PRIV | PS_IG);

            break;

        case TT_TFAULT:

        case TT_TMISS:

        case TT_DFAULT:

        case TT_DMISS:

        case TT_DPROT:

            change_pstate(PS_PEF | PS_PRIV | PS_MG);

            break;

        default:

            change_pstate(PS_PEF | PS_PRIV | PS_AG);

            break;

        }

    }

    if (intno == TT_CLRWIN)

        cpu_set_cwp(env, cpu_cwp_dec(env, env->cwp - 1));

    else if ((intno & 0x1c0) == TT_SPILL)

        cpu_set_cwp(env, cpu_cwp_dec(env, env->cwp - env->cansave - 2));

    else if ((intno & 0x1c0) == TT_FILL)

        cpu_set_cwp(env, cpu_cwp_inc(env, env->cwp + 1));

    env->tbr &= ~0x7fffULL;

    env->tbr |= ((env->tl > 1) ? 1 << 14 : 0) | (intno << 5);

    env->pc = env->tbr;

    env->npc = env->pc + 4;

    env->exception_index = 0;

}
