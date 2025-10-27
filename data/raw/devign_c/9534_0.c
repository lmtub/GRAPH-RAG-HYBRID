static void pc_compat_1_6(MachineState *machine)

{

    pc_compat_1_7(machine);

    rom_file_has_mr = false;

    has_acpi_build = false;

}
