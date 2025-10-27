static void init_proc_620 (CPUPPCState *env)

{

    gen_spr_ne_601(env);

    gen_spr_620(env);

    /* Time base */

    gen_tbl(env);

    /* Hardware implementation registers */

    /* XXX : not implemented */

    spr_register(env, SPR_HID0, "HID0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Memory management */

    gen_low_BATs(env);

    gen_high_BATs(env);

    init_excp_620(env);

    env->dcache_line_size = 64;

    env->icache_line_size = 64;

    /* XXX: TODO: initialize internal interrupt controller */

}
