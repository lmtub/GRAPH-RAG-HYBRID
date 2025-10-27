uint32_t HELPER(get_r13_banked)(CPUState *env, uint32_t mode)

{

    return env->banked_r13[bank_number(mode)];

}
