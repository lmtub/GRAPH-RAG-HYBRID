static void acpi_dsdt_add_cpus(Aml *scope, int smp_cpus)

{

    uint16_t i;



    for (i = 0; i < smp_cpus; i++) {

        Aml *dev = aml_device("C%03x", i);

        aml_append(dev, aml_name_decl("_HID", aml_string("ACPI0007")));

        aml_append(dev, aml_name_decl("_UID", aml_int(i)));

        aml_append(scope, dev);

    }

}
