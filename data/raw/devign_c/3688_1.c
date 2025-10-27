static void spapr_populate_pa_features(CPUPPCState *env, void *fdt, int offset)

{

    uint8_t pa_features_206[] = { 6, 0,

        0xf6, 0x1f, 0xc7, 0x00, 0x80, 0xc0 };

    uint8_t pa_features_207[] = { 24, 0,

        0xf6, 0x1f, 0xc7, 0xc0, 0x80, 0xf0,

        0x80, 0x00, 0x00, 0x00, 0x00, 0x00,

        0x00, 0x00, 0x00, 0x00, 0x80, 0x00,

        0x80, 0x00, 0x80, 0x00, 0x00, 0x00 };

    uint8_t pa_features_300[] = { 66, 0,

        /* 0: MMU|FPU|SLB|RUN|DABR|NX, 1: fri[nzpm]|DABRX|SPRG3|SLB0|PP110 */

        /* 2: VPM|DS205|PPR|DS202|DS206, 3: LSD|URG, SSO, 5: LE|CFAR|EB|LSQ */

        0xf6, 0x1f, 0xc7, 0xc0, 0x80, 0xf0, /* 0 - 5 */

        /* 6: DS207 */

        0x80, 0x00, 0x00, 0x00, 0x00, 0x00, /* 6 - 11 */

        /* 16: Vector */

        0x00, 0x00, 0x00, 0x00, 0x80, 0x00, /* 12 - 17 */

        /* 18: Vec. Scalar, 20: Vec. XOR, 22: HTM */

        0x80, 0x00, 0x80, 0x00, 0x80, 0x00, /* 18 - 23 */

        /* 24: Ext. Dec, 26: 64 bit ftrs, 28: PM ftrs */

        0x80, 0x00, 0x80, 0x00, 0x80, 0x00, /* 24 - 29 */

        /* 30: MMR, 32: LE atomic, 34: EBB + ext EBB */

        0x80, 0x00, 0x80, 0x00, 0xC0, 0x00, /* 30 - 35 */

        /* 36: SPR SO, 38: Copy/Paste, 40: Radix MMU */

        0x80, 0x00, 0x80, 0x00, 0x80, 0x00, /* 36 - 41 */

        /* 42: PM, 44: PC RA, 46: SC vec'd */

        0x80, 0x00, 0x80, 0x00, 0x80, 0x00, /* 42 - 47 */

        /* 48: SIMD, 50: QP BFP, 52: String */

        0x80, 0x00, 0x80, 0x00, 0x80, 0x00, /* 48 - 53 */

        /* 54: DecFP, 56: DecI, 58: SHA */

        0x80, 0x00, 0x80, 0x00, 0x80, 0x00, /* 54 - 59 */

        /* 60: NM atomic, 62: RNG */

        0x80, 0x00, 0x80, 0x00, 0x00, 0x00, /* 60 - 65 */

    };

    uint8_t *pa_features;

    size_t pa_size;



    switch (POWERPC_MMU_VER(env->mmu_model)) {

    case POWERPC_MMU_VER_2_06:

        pa_features = pa_features_206;

        pa_size = sizeof(pa_features_206);

        break;

    case POWERPC_MMU_VER_2_07:

        pa_features = pa_features_207;

        pa_size = sizeof(pa_features_207);

        break;

    case POWERPC_MMU_VER_3_00:

        pa_features = pa_features_300;

        pa_size = sizeof(pa_features_300);

        break;

    default:

        return;

    }



    if (env->ci_large_pages) {

        /*

         * Note: we keep CI large pages off by default because a 64K capable

         * guest provisioned with large pages might otherwise try to map a qemu

         * framebuffer (or other kind of memory mapped PCI BAR) using 64K pages

         * even if that qemu runs on a 4k host.

         * We dd this bit back here if we are confident this is not an issue

         */

        pa_features[3] |= 0x20;

    }

    if (kvmppc_has_cap_htm() && pa_size > 24) {

        pa_features[24] |= 0x80;    /* Transactional memory support */

    }



    _FDT((fdt_setprop(fdt, offset, "ibm,pa-features", pa_features, pa_size)));

}
