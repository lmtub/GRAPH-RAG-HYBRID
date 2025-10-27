static bool enforce_config_section(void)

{

    MachineState *machine = MACHINE(qdev_get_machine());

    return machine->enforce_config_section;

}
