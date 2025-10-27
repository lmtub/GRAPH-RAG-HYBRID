static IsoBcSection *find_iso_bc_entry(void)

{

    IsoBcEntry *e = (IsoBcEntry *)sec;

    uint32_t offset = find_iso_bc();

    int i;



    if (!offset) {

        return NULL;

    }



    read_iso_sector(offset, sec, "Failed to read El Torito boot catalog");



    if (!is_iso_bc_valid(e)) {

        /* The validation entry is mandatory */

        virtio_panic("No valid boot catalog found!\n");

        return NULL;

    }



    /*

     * Each entry has 32 bytes size, so one sector cannot contain > 64 entries.

     * We consider only boot catalogs with no more than 64 entries.

     */

    for (i = 1; i < ISO_BC_ENTRY_PER_SECTOR; i++) {

        if (e[i].id == ISO_BC_BOOTABLE_SECTION) {

            if (is_iso_bc_entry_compatible(&e[i].body.sect)) {

                return &e[i].body.sect;

            }

        }

    }



    virtio_panic("No suitable boot entry found on ISO-9660 media!\n");



    return NULL;

}
