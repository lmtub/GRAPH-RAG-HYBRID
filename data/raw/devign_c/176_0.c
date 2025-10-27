void cpu_x86_init_mmu(CPUX86State *env)

{

    a20_enabled = 1;

    a20_mask = 0xffffffff;



    last_pg_state = -1;

    cpu_x86_update_cr0(env);

}
