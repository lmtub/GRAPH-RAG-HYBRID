void ppc40x_core_reset (CPUState *env)

{

    target_ulong dbsr;



    printf("Reset PowerPC core\n");

    cpu_ppc_reset(env);

    dbsr = env->spr[SPR_40x_DBSR];

    dbsr &= ~0x00000300;

    dbsr |= 0x00000100;

    env->spr[SPR_40x_DBSR] = dbsr;

    cpu_loop_exit();

}
