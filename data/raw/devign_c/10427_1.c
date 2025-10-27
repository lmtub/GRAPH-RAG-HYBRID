int kvm_arch_process_async_events(CPUState *env)
{
    if (kvm_irqchip_in_kernel()) {
    if (env->interrupt_request & (CPU_INTERRUPT_HARD | CPU_INTERRUPT_NMI)) {
    if (env->interrupt_request & CPU_INTERRUPT_INIT) {
        do_cpu_init(env);
    if (env->interrupt_request & CPU_INTERRUPT_SIPI) {
        do_cpu_sipi(env);
    return env->halted;