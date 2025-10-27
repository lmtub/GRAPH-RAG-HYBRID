bool check_hw_breakpoints(CPUX86State *env, bool force_dr6_update)

{

    target_ulong dr6;

    int reg;

    bool hit_enabled = false;



    dr6 = env->dr[6] & ~0xf;

    for (reg = 0; reg < DR7_MAX_BP; reg++) {

        bool bp_match = false;

        bool wp_match = false;



        switch (hw_breakpoint_type(env->dr[7], reg)) {

        case DR7_TYPE_BP_INST:

            if (env->dr[reg] == env->eip) {

                bp_match = true;

            }

            break;

        case DR7_TYPE_DATA_WR:

        case DR7_TYPE_DATA_RW:

            if (env->cpu_watchpoint[reg] &&

                env->cpu_watchpoint[reg]->flags & BP_WATCHPOINT_HIT) {

                wp_match = true;

            }

            break;

        case DR7_TYPE_IO_RW:

            break;

        }

        if (bp_match || wp_match) {

            dr6 |= 1 << reg;

            if (hw_breakpoint_enabled(env->dr[7], reg)) {

                hit_enabled = true;

            }

        }

    }



    if (hit_enabled || force_dr6_update) {

        env->dr[6] = dr6;

    }



    return hit_enabled;

}
