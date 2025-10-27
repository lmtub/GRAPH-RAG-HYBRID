static void do_program_interrupt(CPUS390XState *env)

{

    uint64_t mask, addr;

    LowCore *lowcore;

    hwaddr len = TARGET_PAGE_SIZE;

    int ilen = env->int_pgm_ilen;



    switch (ilen) {

    case ILEN_LATER:

        ilen = get_ilen(cpu_ldub_code(env, env->psw.addr));

        break;

    case ILEN_LATER_INC:

        ilen = get_ilen(cpu_ldub_code(env, env->psw.addr));

        env->psw.addr += ilen;

        break;

    default:

        assert(ilen == 2 || ilen == 4 || ilen == 6);

    }



    qemu_log_mask(CPU_LOG_INT, "%s: code=0x%x ilen=%d\n",

                  __func__, env->int_pgm_code, ilen);



    lowcore = cpu_physical_memory_map(env->psa, &len, 1);



    lowcore->pgm_ilen = cpu_to_be16(ilen);

    lowcore->pgm_code = cpu_to_be16(env->int_pgm_code);

    lowcore->program_old_psw.mask = cpu_to_be64(get_psw_mask(env));

    lowcore->program_old_psw.addr = cpu_to_be64(env->psw.addr);

    mask = be64_to_cpu(lowcore->program_new_psw.mask);

    addr = be64_to_cpu(lowcore->program_new_psw.addr);



    cpu_physical_memory_unmap(lowcore, len, 1, len);



    DPRINTF("%s: %x %x %" PRIx64 " %" PRIx64 "\n", __func__,

            env->int_pgm_code, ilen, env->psw.mask,

            env->psw.addr);



    load_psw(env, mask, addr);

}
