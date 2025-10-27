static int piix3_post_load(void *opaque, int version_id)

{

    PIIX3State *piix3 = opaque;

    int pirq;



    /* Because the i8259 has not been deserialized yet, qemu_irq_raise

     * might bring the system to a different state than the saved one;

     * for example, the interrupt could be masked but the i8259 would

     * not know that yet and would trigger an interrupt in the CPU.

     *

     * Here, we update irq levels without raising the interrupt.

     * Interrupt state will be deserialized separately through the i8259.

     */

    piix3->pic_levels = 0;

    for (pirq = 0; pirq < PIIX_NUM_PIRQS; pirq++) {

        piix3_set_irq_level_internal(piix3, pirq,

                            pci_bus_get_irq_level(piix3->dev.bus, pirq));

    }

    return 0;

}
