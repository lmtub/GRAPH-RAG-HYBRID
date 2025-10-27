static inline void _cpu_ppc_store_hdecr(PowerPCCPU *cpu, uint32_t hdecr,

                                        uint32_t value, int is_excp)

{

    ppc_tb_t *tb_env = cpu->env.tb_env;



    if (tb_env->hdecr_timer != NULL) {

        __cpu_ppc_store_decr(cpu, &tb_env->hdecr_next, tb_env->hdecr_timer,

                             &cpu_ppc_hdecr_excp, hdecr, value, is_excp);

    }

}
