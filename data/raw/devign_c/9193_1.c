int pvpanic_init(ISABus *bus)

{

    isa_create_simple(bus, TYPE_ISA_PVPANIC_DEVICE);

    return 0;

}
