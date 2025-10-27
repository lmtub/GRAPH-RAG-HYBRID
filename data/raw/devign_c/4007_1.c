void hmp_info_cpus(Monitor *mon, const QDict *qdict)

{

    CpuInfoList *cpu_list, *cpu;



    cpu_list = qmp_query_cpus(NULL);



    for (cpu = cpu_list; cpu; cpu = cpu->next) {

        int active = ' ';



        if (cpu->value->CPU == monitor_get_cpu_index()) {

            active = '*';

        }



        monitor_printf(mon, "%c CPU #%" PRId64 ":", active, cpu->value->CPU);



        switch (cpu->value->arch) {

        case CPU_INFO_ARCH_X86:

            monitor_printf(mon, " pc=0x%016" PRIx64, cpu->value->u.x86->pc);

            break;

        case CPU_INFO_ARCH_PPC:

            monitor_printf(mon, " nip=0x%016" PRIx64, cpu->value->u.ppc->nip);

            break;

        case CPU_INFO_ARCH_SPARC:

            monitor_printf(mon, " pc=0x%016" PRIx64,

                           cpu->value->u.q_sparc->pc);

            monitor_printf(mon, " npc=0x%016" PRIx64,

                           cpu->value->u.q_sparc->npc);

            break;

        case CPU_INFO_ARCH_MIPS:

            monitor_printf(mon, " PC=0x%016" PRIx64, cpu->value->u.q_mips->PC);

            break;

        case CPU_INFO_ARCH_TRICORE:

            monitor_printf(mon, " PC=0x%016" PRIx64, cpu->value->u.tricore->PC);

            break;

        default:

            break;

        }



        if (cpu->value->halted) {

            monitor_printf(mon, " (halted)");

        }



        monitor_printf(mon, " thread_id=%" PRId64 "\n", cpu->value->thread_id);

    }



    qapi_free_CpuInfoList(cpu_list);

}
