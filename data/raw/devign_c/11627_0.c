static int cpu_common_load(QEMUFile *f, void *opaque, int version_id)

{

    CPUState *env = opaque;



    if (version_id != CPU_COMMON_SAVE_VERSION)

        return -EINVAL;



    qemu_get_be32s(f, &env->halted);

    qemu_get_be32s(f, &env->interrupt_request);

    env->interrupt_request &= ~CPU_INTERRUPT_EXIT;

    tlb_flush(env, 1);



    return 0;

}
