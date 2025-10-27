static void cpu_ppc_decr_cb(void *opaque)

{

    PowerPCCPU *cpu = opaque;



    _cpu_ppc_store_decr(cpu, 0x00000000, 0xFFFFFFFF, 1);

}
