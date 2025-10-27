void tcg_cpu_address_space_init(CPUState *cpu, AddressSpace *as)

{

    /* We only support one address space per cpu at the moment.  */

    assert(cpu->as == as);



    if (cpu->cpu_ases) {

        /* We've already registered the listener for our only AS */

        return;

    }



    cpu->cpu_ases = g_new0(CPUAddressSpace, 1);

    cpu->cpu_ases[0].cpu = cpu;

    cpu->cpu_ases[0].as = as;

    cpu->cpu_ases[0].tcg_as_listener.commit = tcg_commit;

    memory_listener_register(&cpu->cpu_ases[0].tcg_as_listener, as);

}
