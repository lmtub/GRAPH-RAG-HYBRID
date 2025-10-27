static void ide_cfata_identify(IDEState *s)

{

    uint16_t *p;

    uint32_t cur_sec;



    p = (uint16_t *) s->identify_data;

    if (s->identify_set)

        goto fill_buffer;



    memset(p, 0, sizeof(s->identify_data));



    cur_sec = s->cylinders * s->heads * s->sectors;



    put_le16(p + 0, 0x848a);			/* CF Storage Card signature */

    put_le16(p + 1, s->cylinders);		/* Default cylinders */

    put_le16(p + 3, s->heads);			/* Default heads */

    put_le16(p + 6, s->sectors);		/* Default sectors per track */

    put_le16(p + 7, s->nb_sectors >> 16);	/* Sectors per card */

    put_le16(p + 8, s->nb_sectors);		/* Sectors per card */

    padstr((char *)(p + 10), s->drive_serial_str, 20); /* serial number */

    put_le16(p + 22, 0x0004);			/* ECC bytes */

    padstr((char *) (p + 23), s->version, 8);	/* Firmware Revision */

    padstr((char *) (p + 27), "QEMU MICRODRIVE", 40);/* Model number */

#if MAX_MULT_SECTORS > 1

    put_le16(p + 47, 0x8000 | MAX_MULT_SECTORS);

#else

    put_le16(p + 47, 0x0000);

#endif

    put_le16(p + 49, 0x0f00);			/* Capabilities */

    put_le16(p + 51, 0x0002);			/* PIO cycle timing mode */

    put_le16(p + 52, 0x0001);			/* DMA cycle timing mode */

    put_le16(p + 53, 0x0003);			/* Translation params valid */

    put_le16(p + 54, s->cylinders);		/* Current cylinders */

    put_le16(p + 55, s->heads);			/* Current heads */

    put_le16(p + 56, s->sectors);		/* Current sectors */

    put_le16(p + 57, cur_sec);			/* Current capacity */

    put_le16(p + 58, cur_sec >> 16);		/* Current capacity */

    if (s->mult_sectors)			/* Multiple sector setting */

        put_le16(p + 59, 0x100 | s->mult_sectors);

    put_le16(p + 60, s->nb_sectors);		/* Total LBA sectors */

    put_le16(p + 61, s->nb_sectors >> 16);	/* Total LBA sectors */

    put_le16(p + 63, 0x0203);			/* Multiword DMA capability */

    put_le16(p + 64, 0x0001);			/* Flow Control PIO support */

    put_le16(p + 65, 0x0096);			/* Min. Multiword DMA cycle */

    put_le16(p + 66, 0x0096);			/* Rec. Multiword DMA cycle */

    put_le16(p + 68, 0x00b4);			/* Min. PIO cycle time */

    put_le16(p + 82, 0x400c);			/* Command Set supported */

    put_le16(p + 83, 0x7068);			/* Command Set supported */

    put_le16(p + 84, 0x4000);			/* Features supported */

    put_le16(p + 85, 0x000c);			/* Command Set enabled */

    put_le16(p + 86, 0x7044);			/* Command Set enabled */

    put_le16(p + 87, 0x4000);			/* Features enabled */

    put_le16(p + 91, 0x4060);			/* Current APM level */

    put_le16(p + 129, 0x0002);			/* Current features option */

    put_le16(p + 130, 0x0005);			/* Reassigned sectors */

    put_le16(p + 131, 0x0001);			/* Initial power mode */

    put_le16(p + 132, 0x0000);			/* User signature */

    put_le16(p + 160, 0x8100);			/* Power requirement */

    put_le16(p + 161, 0x8001);			/* CF command set */



    s->identify_set = 1;



fill_buffer:

    memcpy(s->io_buffer, p, sizeof(s->identify_data));

}
