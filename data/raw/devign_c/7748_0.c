static void fdt_add_psci_node(const VirtBoardInfo *vbi)

{

    void *fdt = vbi->fdt;

    ARMCPU *armcpu = ARM_CPU(qemu_get_cpu(0));



    /* No PSCI for TCG yet */

    if (kvm_enabled()) {

        qemu_fdt_add_subnode(fdt, "/psci");

        if (armcpu->psci_version == 2) {

            const char comp[] = "arm,psci-0.2\0arm,psci";

            qemu_fdt_setprop(fdt, "/psci", "compatible", comp, sizeof(comp));

        } else {

            qemu_fdt_setprop_string(fdt, "/psci", "compatible", "arm,psci");

        }



        qemu_fdt_setprop_string(fdt, "/psci", "method", "hvc");

        qemu_fdt_setprop_cell(fdt, "/psci", "cpu_suspend",

                                  QEMU_PSCI_0_1_FN_CPU_SUSPEND);

        qemu_fdt_setprop_cell(fdt, "/psci", "cpu_off", QEMU_PSCI_0_1_FN_CPU_OFF);

        qemu_fdt_setprop_cell(fdt, "/psci", "cpu_on", QEMU_PSCI_0_1_FN_CPU_ON);

        qemu_fdt_setprop_cell(fdt, "/psci", "migrate", QEMU_PSCI_0_1_FN_MIGRATE);

    }

}
