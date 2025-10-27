static void qxl_set_mode(PCIQXLDevice *d, int modenr, int loadvm)

{

    pcibus_t start = d->pci.io_regions[QXL_RAM_RANGE_INDEX].addr;

    pcibus_t end   = d->pci.io_regions[QXL_RAM_RANGE_INDEX].size + start;

    QXLMode *mode = d->modes->modes + modenr;

    uint64_t devmem = d->pci.io_regions[QXL_RAM_RANGE_INDEX].addr;

    QXLMemSlot slot = {

        .mem_start = start,

        .mem_end = end

    };

    QXLSurfaceCreate surface = {

        .width      = mode->x_res,

        .height     = mode->y_res,

        .stride     = -mode->x_res * 4,

        .format     = SPICE_SURFACE_FMT_32_xRGB,

        .flags      = loadvm ? QXL_SURF_FLAG_KEEP_DATA : 0,

        .mouse_mode = true,

        .mem        = devmem + d->shadow_rom.draw_area_offset,

    };



    trace_qxl_set_mode(d->id, modenr, mode->x_res, mode->y_res, mode->bits,

                       devmem);

    if (!loadvm) {

        qxl_hard_reset(d, 0);

    }



    d->guest_slots[0].slot = slot;

    qxl_add_memslot(d, 0, devmem, QXL_SYNC);



    d->guest_primary.surface = surface;

    qxl_create_guest_primary(d, 0, QXL_SYNC);



    d->mode = QXL_MODE_COMPAT;

    d->cmdflags = QXL_COMMAND_FLAG_COMPAT;

#ifdef QXL_COMMAND_FLAG_COMPAT_16BPP /* new in spice 0.6.1 */

    if (mode->bits == 16) {

        d->cmdflags |= QXL_COMMAND_FLAG_COMPAT_16BPP;

    }

#endif

    d->shadow_rom.mode = cpu_to_le32(modenr);

    d->rom->mode = cpu_to_le32(modenr);

    qxl_rom_set_dirty(d);

}
