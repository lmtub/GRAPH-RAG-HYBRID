static void test_acpi_piix4_tcg_cphp(void)

{

    test_data data;



    memset(&data, 0, sizeof(data));

    data.machine = MACHINE_PC;

    data.variant = ".cphp";

    test_acpi_one("-smp 2,cores=3,sockets=2,maxcpus=6",

                  &data);

    free_test_data(&data);

}
