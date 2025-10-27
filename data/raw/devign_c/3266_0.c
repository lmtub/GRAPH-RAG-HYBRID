static void patch_pcihp(int slot, uint8_t *ssdt_ptr, uint32_t eject)

{

    ssdt_ptr[ACPI_PCIHP_OFFSET_HEX] = acpi_get_hex(slot >> 4);

    ssdt_ptr[ACPI_PCIHP_OFFSET_HEX + 1] = acpi_get_hex(slot);

    ssdt_ptr[ACPI_PCIHP_OFFSET_ID] = slot;

    ssdt_ptr[ACPI_PCIHP_OFFSET_ADR + 2] = slot;



    /* Runtime patching of ACPI_EJ0: to disable hotplug for a slot,

     * replace the method name: _EJ0 by ACPI_EJ0_.

     */

    /* Sanity check */

    assert(!memcmp(ssdt_ptr + ACPI_PCIHP_OFFSET_EJ0, "_EJ0", 4));



    if (!eject) {

        memcpy(ssdt_ptr + ACPI_PCIHP_OFFSET_EJ0, "EJ0_", 4);

    }

}
