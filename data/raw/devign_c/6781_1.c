static void pc_init_pci_1_4(QEMUMachineInitArgs *args)

{

    pc_sysfw_flash_vs_rom_bug_compatible = true;

    has_pvpanic = false;

    x86_cpu_compat_set_features("n270", FEAT_1_ECX, 0, CPUID_EXT_MOVBE);

    pc_init_pci(args);

}
