static void create_cps(MaltaState *s, const char *cpu_model,

                       qemu_irq *cbus_irq, qemu_irq *i8259_irq)

{

    Error *err = NULL;

    s->cps = g_new0(MIPSCPSState, 1);



    object_initialize(s->cps, sizeof(MIPSCPSState), TYPE_MIPS_CPS);

    qdev_set_parent_bus(DEVICE(s->cps), sysbus_get_default());



    object_property_set_str(OBJECT(s->cps), cpu_model, "cpu-model", &err);

    object_property_set_int(OBJECT(s->cps), smp_cpus, "num-vp", &err);

    object_property_set_bool(OBJECT(s->cps), true, "realized", &err);

    if (err != NULL) {

        error_report("%s", error_get_pretty(err));

        exit(1);

    }



    sysbus_mmio_map_overlap(SYS_BUS_DEVICE(s->cps), 0, 0, 1);



    /* FIXME: When GIC is present then we should use GIC's IRQ 3.

       Until then CPS exposes CPU's IRQs thus use the default IRQ 2. */

    *i8259_irq = get_cps_irq(s->cps, 2);

    *cbus_irq = NULL;

}
