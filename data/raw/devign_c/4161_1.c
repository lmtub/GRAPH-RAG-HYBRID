static void tlb_flush_nocheck(CPUState *cpu)

{

    CPUArchState *env = cpu->env_ptr;



    /* The QOM tests will trigger tlb_flushes without setting up TCG

     * so we bug out here in that case.

     */

    if (!tcg_enabled()) {

        return;

    }



    assert_cpu_is_self(cpu);

    tlb_debug("(count: %d)\n", tlb_flush_count++);



    tb_lock();



    memset(env->tlb_table, -1, sizeof(env->tlb_table));

    memset(env->tlb_v_table, -1, sizeof(env->tlb_v_table));

    memset(cpu->tb_jmp_cache, 0, sizeof(cpu->tb_jmp_cache));



    env->vtlb_index = 0;

    env->tlb_flush_addr = -1;

    env->tlb_flush_mask = 0;



    tb_unlock();



    atomic_mb_set(&cpu->pending_tlb_flush, 0);

}
