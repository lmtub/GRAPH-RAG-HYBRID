static void vexpress_modify_dtb(const struct arm_boot_info *info, void *fdt)

{

    uint32_t acells, scells, intc;

    const VEDBoardInfo *daughterboard = (const VEDBoardInfo *)info;



    acells = qemu_fdt_getprop_cell(fdt, "/", "#address-cells",

                                   NULL, &error_fatal);

    scells = qemu_fdt_getprop_cell(fdt, "/", "#size-cells",

                                   NULL, &error_fatal);

    intc = find_int_controller(fdt);

    if (!intc) {

        /* Not fatal, we just won't provide virtio. This will

         * happen with older device tree blobs.

         */

        fprintf(stderr, "QEMU: warning: couldn't find interrupt controller in "

                "dtb; will not include virtio-mmio devices in the dtb.\n");

    } else {

        int i;

        const hwaddr *map = daughterboard->motherboard_map;



        /* We iterate backwards here because adding nodes

         * to the dtb puts them in last-first.

         */

        for (i = NUM_VIRTIO_TRANSPORTS - 1; i >= 0; i--) {

            add_virtio_mmio_node(fdt, acells, scells,

                                 map[VE_VIRTIO] + 0x200 * i,

                                 0x200, intc, 40 + i);

        }

    }

}
