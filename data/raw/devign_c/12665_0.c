void HELPER(set_r13_banked)(CPUState *env, uint32_t mode, uint32_t val)

{

    env->banked_r13[bank_number(mode)] = val;

}
