int cpu_is_bsp(CPUState *env)

{

    return env->cpuid_apic_id == 0;

}
