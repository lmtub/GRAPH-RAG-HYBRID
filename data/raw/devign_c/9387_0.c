void tb_check_watchpoint(CPUState *cpu)

{

    TranslationBlock *tb;



    tb = tb_find_pc(cpu->mem_io_pc);

    if (!tb) {

        cpu_abort(cpu, "check_watchpoint: could not find TB for pc=%p",

                  (void *)cpu->mem_io_pc);

    }

    cpu_restore_state_from_tb(cpu, tb, cpu->mem_io_pc);

    tb_phys_invalidate(tb, -1);

}
