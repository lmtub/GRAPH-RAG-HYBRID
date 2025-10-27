static void *acpi_set_bsel(PCIBus *bus, void *opaque)

{

    unsigned *bsel_alloc = opaque;

    unsigned *bus_bsel;



    if (qbus_is_hotpluggable(BUS(bus))) {

        bus_bsel = g_malloc(sizeof *bus_bsel);



        *bus_bsel = (*bsel_alloc)++;

        object_property_add_uint32_ptr(OBJECT(bus), ACPI_PCIHP_PROP_BSEL,

                                       bus_bsel, NULL);

    }



    return bsel_alloc;

}
