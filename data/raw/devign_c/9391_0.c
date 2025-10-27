static void fdt_add_cpu_nodes(const VirtBoardInfo *vbi)

{

    int cpu;

    int addr_cells = 1;



    /*

     * From Documentation/devicetree/bindings/arm/cpus.txt

     *  On ARM v8 64-bit systems value should be set to 2,

     *  that corresponds to the MPIDR_EL1 register size.

     *  If MPIDR_EL1[63:32] value is equal to 0 on all CPUs

     *  in the system, #address-cells can be set to 1, since

     *  MPIDR_EL1[63:32] bits are not used for CPUs

     *  identification.

     *

     *  Here we actually don't know whether our system is 32- or 64-bit one.

     *  The simplest way to go is to examine affinity IDs of all our CPUs. If

     *  at least one of them has Aff3 populated, we set #address-cells to 2.

     */

    for (cpu = 0; cpu < vbi->smp_cpus; cpu++) {

        ARMCPU *armcpu = ARM_CPU(qemu_get_cpu(cpu));



        if (armcpu->mp_affinity & ARM_AFF3_MASK) {

            addr_cells = 2;

            break;

        }

    }



    qemu_fdt_add_subnode(vbi->fdt, "/cpus");

    qemu_fdt_setprop_cell(vbi->fdt, "/cpus", "#address-cells", addr_cells);

    qemu_fdt_setprop_cell(vbi->fdt, "/cpus", "#size-cells", 0x0);



    for (cpu = vbi->smp_cpus - 1; cpu >= 0; cpu--) {

        char *nodename = g_strdup_printf("/cpus/cpu@%d", cpu);

        ARMCPU *armcpu = ARM_CPU(qemu_get_cpu(cpu));



        qemu_fdt_add_subnode(vbi->fdt, nodename);

        qemu_fdt_setprop_string(vbi->fdt, nodename, "device_type", "cpu");

        qemu_fdt_setprop_string(vbi->fdt, nodename, "compatible",

                                    armcpu->dtb_compatible);



        if (vbi->smp_cpus > 1) {

            qemu_fdt_setprop_string(vbi->fdt, nodename,

                                        "enable-method", "psci");

        }



        if (addr_cells == 2) {

            qemu_fdt_setprop_u64(vbi->fdt, nodename, "reg",

                                 armcpu->mp_affinity);

        } else {

            qemu_fdt_setprop_cell(vbi->fdt, nodename, "reg",

                                  armcpu->mp_affinity);

        }



        g_free(nodename);

    }

}
