static int vga_load(QEMUFile *f, void *opaque, int version_id)

{

    VGAState *s = opaque;

    int is_vbe, i, ret;



    if (version_id > 2)

        return -EINVAL;



    if (s->pci_dev && version_id >= 2) {

        ret = pci_device_load(s->pci_dev, f);

        if (ret < 0)

            return ret;

    }



    qemu_get_be32s(f, &s->latch);

    qemu_get_8s(f, &s->sr_index);

    qemu_get_buffer(f, s->sr, 8);

    qemu_get_8s(f, &s->gr_index);

    qemu_get_buffer(f, s->gr, 16);

    qemu_get_8s(f, &s->ar_index);

    qemu_get_buffer(f, s->ar, 21);

    s->ar_flip_flop=qemu_get_be32(f);

    qemu_get_8s(f, &s->cr_index);

    qemu_get_buffer(f, s->cr, 256);

    qemu_get_8s(f, &s->msr);

    qemu_get_8s(f, &s->fcr);

    qemu_get_8s(f, &s->st00);

    qemu_get_8s(f, &s->st01);



    qemu_get_8s(f, &s->dac_state);

    qemu_get_8s(f, &s->dac_sub_index);

    qemu_get_8s(f, &s->dac_read_index);

    qemu_get_8s(f, &s->dac_write_index);

    qemu_get_buffer(f, s->dac_cache, 3);

    qemu_get_buffer(f, s->palette, 768);



    s->bank_offset=qemu_get_be32(f);

    is_vbe = qemu_get_byte(f);

#ifdef CONFIG_BOCHS_VBE

    if (!is_vbe)

        return -EINVAL;

    qemu_get_be16s(f, &s->vbe_index);

    for(i = 0; i < VBE_DISPI_INDEX_NB; i++)

        qemu_get_be16s(f, &s->vbe_regs[i]);

    qemu_get_be32s(f, &s->vbe_start_addr);

    qemu_get_be32s(f, &s->vbe_line_offset);

    qemu_get_be32s(f, &s->vbe_bank_mask);

#else

    if (is_vbe)

        return -EINVAL;

#endif



    /* force refresh */

    s->graphic_mode = -1;

    return 0;

}
