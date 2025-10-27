target_ulong helper_cfc1(CPUMIPSState *env, uint32_t reg)

{

    target_ulong arg1;



    switch (reg) {

    case 0:

        arg1 = (int32_t)env->active_fpu.fcr0;

        break;

    case 25:

        arg1 = ((env->active_fpu.fcr31 >> 24) & 0xfe) | ((env->active_fpu.fcr31 >> 23) & 0x1);

        break;

    case 26:

        arg1 = env->active_fpu.fcr31 & 0x0003f07c;

        break;

    case 28:

        arg1 = (env->active_fpu.fcr31 & 0x00000f83) | ((env->active_fpu.fcr31 >> 22) & 0x4);

        break;

    default:

        arg1 = (int32_t)env->active_fpu.fcr31;

        break;

    }



    return arg1;

}
