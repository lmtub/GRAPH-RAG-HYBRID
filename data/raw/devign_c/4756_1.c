void HELPER(cpsr_write_eret)(CPUARMState *env, uint32_t val)
{
    cpsr_write(env, val, CPSR_ERET_MASK, CPSRWriteExceptionReturn);
    arm_call_el_change_hook(arm_env_get_cpu(env));
}