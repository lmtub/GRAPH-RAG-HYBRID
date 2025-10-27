void *HELPER(lookup_tb_ptr)(CPUArchState *env)

{

    CPUState *cpu = ENV_GET_CPU(env);

    TranslationBlock *tb;

    target_ulong cs_base, pc;

    uint32_t flags, hash;



    cpu_get_tb_cpu_state(env, &pc, &cs_base, &flags);

    hash = tb_jmp_cache_hash_func(pc);

    tb = atomic_rcu_read(&cpu->tb_jmp_cache[hash]);



    if (unlikely(!(tb

                   && tb->pc == pc

                   && tb->cs_base == cs_base

                   && tb->flags == flags

                   && tb->trace_vcpu_dstate == *cpu->trace_dstate))) {

        tb = tb_htable_lookup(cpu, pc, cs_base, flags);

        if (!tb) {

            return tcg_ctx.code_gen_epilogue;

        }

        atomic_set(&cpu->tb_jmp_cache[hash], tb);

    }



    qemu_log_mask_and_addr(CPU_LOG_EXEC, pc,

                           "Chain %p [%d: " TARGET_FMT_lx "] %s\n",

                           tb->tc_ptr, cpu->cpu_index, pc,

                           lookup_symbol(pc));

    return tb->tc_ptr;

}
