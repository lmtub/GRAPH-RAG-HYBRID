static void switch_v7m_sp(CPUARMState *env, bool new_spsel)

{

    uint32_t tmp;

    bool old_spsel = env->v7m.control & R_V7M_CONTROL_SPSEL_MASK;



    if (old_spsel != new_spsel) {

        tmp = env->v7m.other_sp;

        env->v7m.other_sp = env->regs[13];

        env->regs[13] = tmp;



        env->v7m.control = deposit32(env->v7m.control,

                                     R_V7M_CONTROL_SPSEL_SHIFT,

                                     R_V7M_CONTROL_SPSEL_LENGTH, new_spsel);

    }

}
