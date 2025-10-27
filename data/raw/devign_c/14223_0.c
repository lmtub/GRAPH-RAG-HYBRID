static void gen_spr_405 (CPUPPCState *env)

{

    spr_register(env, SPR_4xx_CCR0, "CCR0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00700000);

    /* Debug */

    /* XXX : not implemented */

    spr_register(env, SPR_405_DBCR1, "DBCR1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_405_DVC1, "DVC1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_405_DVC2, "DVC2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_405_IAC3, "IAC3",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_405_IAC4, "IAC4",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Storage control */

    spr_register(env, SPR_405_SLER, "SLER",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_40x_sler,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_405_SU0R, "SU0R",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* SPRG */

    spr_register(env, SPR_USPRG0, "USPRG0",

                 &spr_read_ureg, SPR_NOACCESS,

                 &spr_read_ureg, SPR_NOACCESS,

                 0x00000000);

    spr_register(env, SPR_SPRG4, "SPRG4",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_USPRG4, "USPRG4",

                 &spr_read_ureg, SPR_NOACCESS,

                 &spr_read_ureg, SPR_NOACCESS,

                 0x00000000);

    spr_register(env, SPR_SPRG5, "SPRG5",

                 SPR_NOACCESS, SPR_NOACCESS,

                 spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_USPRG5, "USPRG5",

                 &spr_read_ureg, SPR_NOACCESS,

                 &spr_read_ureg, SPR_NOACCESS,

                 0x00000000);

    spr_register(env, SPR_SPRG6, "SPRG6",

                 SPR_NOACCESS, SPR_NOACCESS,

                 spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_USPRG6, "USPRG6",

                 &spr_read_ureg, SPR_NOACCESS,

                 &spr_read_ureg, SPR_NOACCESS,

                 0x00000000);

    spr_register(env, SPR_SPRG7, "SPRG7",

                 SPR_NOACCESS, SPR_NOACCESS,

                 spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_USPRG7, "USPRG7",

                 &spr_read_ureg, SPR_NOACCESS,

                 &spr_read_ureg, SPR_NOACCESS,

                 0x00000000);

}
