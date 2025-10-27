void hd_geometry_guess(BlockBackend *blk,

                       uint32_t *pcyls, uint32_t *pheads, uint32_t *psecs,

                       int *ptrans)

{

    int cylinders, heads, secs, translation;



    if (guess_disk_lchs(blk, &cylinders, &heads, &secs) < 0) {

        /* no LCHS guess: use a standard physical disk geometry  */

        guess_chs_for_size(blk, pcyls, pheads, psecs);

        translation = hd_bios_chs_auto_trans(*pcyls, *pheads, *psecs);

    } else if (heads > 16) {

        /* LCHS guess with heads > 16 means that a BIOS LBA

           translation was active, so a standard physical disk

           geometry is OK */

        guess_chs_for_size(blk, pcyls, pheads, psecs);

        translation = *pcyls * *pheads <= 131072

            ? BIOS_ATA_TRANSLATION_LARGE

            : BIOS_ATA_TRANSLATION_LBA;

    } else {

        /* LCHS guess with heads <= 16: use as physical geometry */

        *pcyls = cylinders;

        *pheads = heads;

        *psecs = secs;

        /* disable any translation to be in sync with

           the logical geometry */

        translation = BIOS_ATA_TRANSLATION_NONE;

    }

    if (ptrans) {

        *ptrans = translation;

    }

    trace_hd_geometry_guess(blk, *pcyls, *pheads, *psecs, translation);

}
