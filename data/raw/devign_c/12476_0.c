static void acpi_set_cpu_present_bit(AcpiCpuHotplug *g, CPUState *cpu,

                                     Error **errp)

{

    CPUClass *k = CPU_GET_CLASS(cpu);

    int64_t cpu_id;



    cpu_id = k->get_arch_id(cpu);

    if ((cpu_id / 8) >= ACPI_GPE_PROC_LEN) {

        error_setg(errp, "acpi: invalid cpu id: %" PRIi64, cpu_id);

        return;

    }



    g->sts[cpu_id / 8] |= (1 << (cpu_id % 8));

}
