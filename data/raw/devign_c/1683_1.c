static target_long monitor_get_decr (const struct MonitorDef *md, int val)

{

    CPUState *env = mon_get_cpu();

    if (!env)

        return 0;

    return cpu_ppc_load_decr(env);

}
