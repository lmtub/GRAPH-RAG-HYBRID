int check_hw_breakpoints(CPUX86State *env, int force_dr6_update)

{

    target_ulong dr6;

    int reg, type;

    int hit_enabled = 0;



    dr6 = env->dr[6] & ~0xf;

    for (reg = 0; reg < DR7_MAX_BP; reg++) {

        type = hw_breakpoint_type(env->dr[7], reg);

        if ((type == 0 && env->dr[reg] == env->eip) ||

            ((type & 1) && env->cpu_watchpoint[reg] &&

             (env->cpu_watchpoint[reg]->flags & BP_WATCHPOINT_HIT))) {

            dr6 |= 1 << reg;

            if (hw_breakpoint_enabled(env->dr[7], reg)) {

                hit_enabled = 1;

            }

        }

    }

    if (hit_enabled || force_dr6_update)

        env->dr[6] = dr6;

    return hit_enabled;

}
