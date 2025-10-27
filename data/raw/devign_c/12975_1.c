void AcpiCpuHotplug_add(ACPIGPE *gpe, AcpiCpuHotplug *g, CPUState *cpu)

{

    CPUClass *k = CPU_GET_CLASS(cpu);

    int64_t cpu_id;



    *gpe->sts = *gpe->sts | ACPI_CPU_HOTPLUG_STATUS;

    cpu_id = k->get_arch_id(CPU(cpu));


    g->sts[cpu_id / 8] |= (1 << (cpu_id % 8));

}