static uint32_t arm_v7m_load_vector(ARMCPU *cpu)



{

    CPUState *cs = CPU(cpu);

    CPUARMState *env = &cpu->env;

    MemTxResult result;

    hwaddr vec = env->v7m.vecbase + env->v7m.exception * 4;

    uint32_t addr;



    addr = address_space_ldl(cs->as, vec,

                             MEMTXATTRS_UNSPECIFIED, &result);

    if (result != MEMTX_OK) {

        /* Architecturally this should cause a HardFault setting HSFR.VECTTBL,

         * which would then be immediately followed by our failing to load

         * the entry vector for that HardFault, which is a Lockup case.

         * Since we don't model Lockup, we just report this guest error

         * via cpu_abort().

         */

        cpu_abort(cs, "Failed to read from exception vector table "

                  "entry %08x\n", (unsigned)vec);

    }

    return addr;

}
