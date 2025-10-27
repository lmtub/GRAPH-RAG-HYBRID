static void qxl_realize_common(PCIQXLDevice *qxl, Error **errp)

{

    uint8_t* config = qxl->pci.config;

    uint32_t pci_device_rev;

    uint32_t io_size;



    qxl->mode = QXL_MODE_UNDEFINED;

    qxl->generation = 1;

    qxl->num_memslots = NUM_MEMSLOTS;

    qemu_mutex_init(&qxl->track_lock);

    qemu_mutex_init(&qxl->async_lock);

    qxl->current_async = QXL_UNDEFINED_IO;

    qxl->guest_bug = 0;



    switch (qxl->revision) {

    case 1: /* spice 0.4 -- qxl-1 */

        pci_device_rev = QXL_REVISION_STABLE_V04;

        io_size = 8;

        break;

    case 2: /* spice 0.6 -- qxl-2 */

        pci_device_rev = QXL_REVISION_STABLE_V06;

        io_size = 16;

        break;

    case 3: /* qxl-3 */

        pci_device_rev = QXL_REVISION_STABLE_V10;

        io_size = 32; /* PCI region size must be pow2 */

        break;

    case 4: /* qxl-4 */

        pci_device_rev = QXL_REVISION_STABLE_V12;

        io_size = pow2ceil(QXL_IO_RANGE_SIZE);

        break;

    default:

        error_setg(errp, "Invalid revision %d for qxl device (max %d)",

                   qxl->revision, QXL_DEFAULT_REVISION);

        return;

    }



    pci_set_byte(&config[PCI_REVISION_ID], pci_device_rev);

    pci_set_byte(&config[PCI_INTERRUPT_PIN], 1);



    qxl->rom_size = qxl_rom_size();

    memory_region_init_ram(&qxl->rom_bar, OBJECT(qxl), "qxl.vrom",

                           qxl->rom_size, &error_abort);

    vmstate_register_ram(&qxl->rom_bar, &qxl->pci.qdev);

    init_qxl_rom(qxl);

    init_qxl_ram(qxl);



    qxl->guest_surfaces.cmds = g_new0(QXLPHYSICAL, qxl->ssd.num_surfaces);

    memory_region_init_ram(&qxl->vram_bar, OBJECT(qxl), "qxl.vram",

                           qxl->vram_size, &error_abort);

    vmstate_register_ram(&qxl->vram_bar, &qxl->pci.qdev);

    memory_region_init_alias(&qxl->vram32_bar, OBJECT(qxl), "qxl.vram32",

                             &qxl->vram_bar, 0, qxl->vram32_size);



    memory_region_init_io(&qxl->io_bar, OBJECT(qxl), &qxl_io_ops, qxl,

                          "qxl-ioports", io_size);

    if (qxl->id == 0) {

        vga_dirty_log_start(&qxl->vga);

    }

    memory_region_set_flush_coalesced(&qxl->io_bar);





    pci_register_bar(&qxl->pci, QXL_IO_RANGE_INDEX,

                     PCI_BASE_ADDRESS_SPACE_IO, &qxl->io_bar);



    pci_register_bar(&qxl->pci, QXL_ROM_RANGE_INDEX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &qxl->rom_bar);



    pci_register_bar(&qxl->pci, QXL_RAM_RANGE_INDEX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &qxl->vga.vram);



    pci_register_bar(&qxl->pci, QXL_VRAM_RANGE_INDEX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &qxl->vram32_bar);



    if (qxl->vram32_size < qxl->vram_size) {

        /*

         * Make the 64bit vram bar show up only in case it is

         * configured to be larger than the 32bit vram bar.

         */

        pci_register_bar(&qxl->pci, QXL_VRAM64_RANGE_INDEX,

                         PCI_BASE_ADDRESS_SPACE_MEMORY |

                         PCI_BASE_ADDRESS_MEM_TYPE_64 |

                         PCI_BASE_ADDRESS_MEM_PREFETCH,

                         &qxl->vram_bar);

    }



    /* print pci bar details */

    dprint(qxl, 1, "ram/%s: %d MB [region 0]\n",

           qxl->id == 0 ? "pri" : "sec",

           qxl->vga.vram_size / (1024*1024));

    dprint(qxl, 1, "vram/32: %d MB [region 1]\n",

           qxl->vram32_size / (1024*1024));

    dprint(qxl, 1, "vram/64: %d MB %s\n",

           qxl->vram_size / (1024*1024),

           qxl->vram32_size < qxl->vram_size ? "[region 4]" : "[unmapped]");



    qxl->ssd.qxl.base.sif = &qxl_interface.base;

    if (qemu_spice_add_display_interface(&qxl->ssd.qxl, qxl->vga.con) != 0) {

        error_setg(errp, "qxl interface %d.%d not supported by spice-server",

                   SPICE_INTERFACE_QXL_MAJOR, SPICE_INTERFACE_QXL_MINOR);

        return;

    }

    qemu_add_vm_change_state_handler(qxl_vm_change_state_handler, qxl);



    qxl->update_irq = qemu_bh_new(qxl_update_irq_bh, qxl);

    qxl_reset_state(qxl);



    qxl->update_area_bh = qemu_bh_new(qxl_render_update_area_bh, qxl);

    qxl->ssd.cursor_bh = qemu_bh_new(qemu_spice_cursor_refresh_bh, &qxl->ssd);

}
