static void test_acpi_one(const char *params, test_data *data)

{

    char *args;

    uint8_t signature_low;

    uint8_t signature_high;

    uint16_t signature;

    int i;

    const char *device = "";



    if (!g_strcmp0(data->machine, MACHINE_Q35)) {

        device = ",id=hd -device ide-hd,drive=hd";

    }



    args = g_strdup_printf("-net none -display none %s -drive file=%s%s,",

                           params ? params : "", disk, device);

    qtest_start(args);



   /* Wait at most 1 minute */

#define TEST_DELAY (1 * G_USEC_PER_SEC / 10)

#define TEST_CYCLES MAX((60 * G_USEC_PER_SEC / TEST_DELAY), 1)



    /* Poll until code has run and modified memory.  Once it has we know BIOS

     * initialization is done.  TODO: check that IP reached the halt

     * instruction.

     */

    for (i = 0; i < TEST_CYCLES; ++i) {

        signature_low = readb(BOOT_SECTOR_ADDRESS + SIGNATURE_OFFSET);

        signature_high = readb(BOOT_SECTOR_ADDRESS + SIGNATURE_OFFSET + 1);

        signature = (signature_high << 8) | signature_low;

        if (signature == SIGNATURE) {

            break;

        }

        g_usleep(TEST_DELAY);

    }

    g_assert_cmphex(signature, ==, SIGNATURE);



    test_acpi_rsdp_address(data);

    test_acpi_rsdp_table(data);

    test_acpi_rsdt_table(data);

    test_acpi_fadt_table(data);

    test_acpi_facs_table(data);

    test_acpi_dsdt_table(data);

    test_acpi_tables(data);



    if (iasl) {

        if (getenv(ACPI_REBUILD_EXPECTED_AML)) {

            dump_aml_files(data, true);

        } else {

            test_acpi_asl(data);

        }

    }



    test_smbios_ep_address(data);

    test_smbios_ep_table(data);

    test_smbios_structs(data);



    qtest_quit(global_qtest);

    g_free(args);

}
