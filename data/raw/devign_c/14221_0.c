static void cpu_ppc_hdecr_cb(void *opaque)

{

    PowerPCCPU *cpu = opaque;



    _cpu_ppc_store_hdecr(cpu, 0x00000000, 0xFFFFFFFF, 1);

}
