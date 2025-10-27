static void pc_q35_machine_options(MachineClass *m)

{

    m->family = "pc_q35";

    m->desc = "Standard PC (Q35 + ICH9, 2009)";

    m->hot_add_cpu = pc_hot_add_cpu;

    m->units_per_default_bus = 1;

    m->default_machine_opts = "firmware=bios-256k.bin";

    m->default_display = "std";

    m->no_floppy = 1;


}