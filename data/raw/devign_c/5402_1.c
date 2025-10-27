void apic_reset_irq_delivered(void)

{

    trace_apic_reset_irq_delivered(apic_irq_delivered);



    apic_irq_delivered = 0;

}
