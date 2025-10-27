void isa_bus_irqs(ISABus *bus, qemu_irq *irqs)

{

    if (!bus) {

        hw_error("Can't set isa irqs with no isa bus present.");

    }

    bus->irqs = irqs;

}
