static void tilegx_cpu_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    CPUClass *cc = CPU_CLASS(oc);
    TileGXCPUClass *tcc = TILEGX_CPU_CLASS(oc);
    tcc->parent_realize = dc->realize;
    dc->realize = tilegx_cpu_realizefn;
    tcc->parent_reset = cc->reset;
    cc->reset = tilegx_cpu_reset;
    cc->has_work = tilegx_cpu_has_work;
    cc->do_interrupt = tilegx_cpu_do_interrupt;
    cc->cpu_exec_interrupt = tilegx_cpu_exec_interrupt;
    cc->dump_state = tilegx_cpu_dump_state;
    cc->set_pc = tilegx_cpu_set_pc;
    cc->handle_mmu_fault = tilegx_cpu_handle_mmu_fault;
    cc->gdb_num_core_regs = 0;
}