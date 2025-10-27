static void vmstate_subsection_save(QEMUFile *f, const VMStateDescription *vmsd,

                                    void *opaque)

{

    const VMStateSubsection *sub = vmsd->subsections;



    while (sub && sub->needed) {

        if (sub->needed(opaque)) {

            const VMStateDescription *vmsd = sub->vmsd;

            uint8_t len;



            qemu_put_byte(f, QEMU_VM_SUBSECTION);

            len = strlen(vmsd->name);

            qemu_put_byte(f, len);

            qemu_put_buffer(f, (uint8_t *)vmsd->name, len);

            qemu_put_be32(f, vmsd->version_id);

            assert(!vmsd->subsections);

            vmstate_save_state(f, vmsd, opaque);

        }

        sub++;

    }

}
